import pytest
import numpy as np
import scipy.stats
from math import lgamma, log, exp, sqrt
import time
import numba
from numba import njit
from cnaster.hmm_utils import convert_params
from cnaster.deprecated.hmm_nophasing import
from cnaster.deprecated.hmm_nophasing import compute_emission_probability_nb_betabinom

@njit
def convert_params_numba(mean, std):
    var = std * std
    p = mean / var
    n = mean * p / (1.0 - p)
    return n, p


@njit
def nbinom_logpmf_numba(k, r, p):
    if p <= 0.0 or p >= 1.0 or r <= 0.0:
        return 0.

    if k < 0:
        return 0.

    log_coeff = lgamma(k + r) - lgamma(k + 1) - lgamma(r)
    return log_coeff + r * log(p) + k * log(1.0 - p)


@njit(nopython=True)
def betabinom_logpmf_numba(k, n, alpha, beta):
    if alpha <= 0.0 or beta <= 0.0 or n < 0 or k < 0 or k > n:
        return 0.0

    log_binom_coeff = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
    log_beta_denom = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

    return log_binom_coeff + log_beta_num - log_beta_denom


@njit(parallel=True)
def compute_emissions_numba(
    base_nb_mean,
    log_mu,
    alphas,
    total_bb_RD,
    p_binom,
    taus,
    X,
    n_states,
    n_obs,
    n_spots,
):
    # TODO zeros? -np.inf
    log_emission_rdr = np.full((n_states, n_obs, n_spots), 0.0)
    log_emission_baf = np.full((n_states, n_obs, n_spots), 0.0)

    for i in numba.prange(n_states):
        for s in range(n_spots):
            for obs in range(n_obs):
                if base_nb_mean[obs, s] > 0:
                    nb_mean = base_nb_mean[obs, s] * exp(log_mu[i, s])
                    nb_var = nb_mean + alphas[i, s] * nb_mean * nb_mean
                    nb_std = sqrt(nb_var)

                    n, p = convert_params_numba(nb_mean, nb_std)
                    log_emission_rdr[i, obs, s] = nbinom_logpmf_numba(
                        X[obs, 0, s], n, p
                    )

            for obs in range(n_obs):
                if total_bb_RD[obs, s] > 0:
                    alpha = p_binom[i, s] * taus[i, s]
                    beta = (1.0 - p_binom[i, s]) * taus[i, s]

                    log_emission_baf[i, obs, s] = betabinom_logpmf_numba(
                        X[obs, 1, s], total_bb_RD[obs, s], alpha, beta
                    )

    return log_emission_rdr, log_emission_baf


def compute_emissions_efficient(
    base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
):
    n_obs, n_spots = base_nb_mean.shape
    n_states = log_mu.shape[0]

    base_nb_mean = np.ascontiguousarray(base_nb_mean, dtype=np.float64)
    log_mu = np.ascontiguousarray(log_mu, dtype=np.float64)
    alphas = np.ascontiguousarray(alphas, dtype=np.float64)
    total_bb_RD = np.ascontiguousarray(total_bb_RD, dtype=np.int32)
    p_binom = np.ascontiguousarray(p_binom, dtype=np.float64)
    taus = np.ascontiguousarray(taus, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.int32)

    return compute_emissions_numba(
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        X,
        n_states,
        n_obs,
        n_spots,
    )

@pytest.fixture
def test_data():
    np.random.seed(123)
    n_obs, n_spots, n_states = 4_494, 5, 10

    base_nb_mean = np.random.exponential(2.0, (n_obs, n_spots))

    # NB models dropout
    base_nb_mean[np.random.random((n_obs, n_spots)) < 0.1] = 0

    log_mu = np.random.normal(0, 0.5, (n_states, n_spots))
    alphas = np.random.exponential(0.1, (n_states, n_spots))

    total_bb_RD = np.random.poisson(20, (n_obs, n_spots))
    total_bb_RD[np.random.random((n_obs, n_spots)) < 0.1] = 0

    p_binom = np.random.beta(2, 2, (n_states, n_spots))
    taus = np.random.exponential(10, (n_states, n_spots))

    X = np.zeros((n_obs, 2, n_spots), dtype=int)
    X[:, 0, :] = np.random.poisson(base_nb_mean * 2)
    X[:, 1, :] = np.random.binomial(total_bb_RD, 0.5)

    return base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X

class TestEquivalence:
    def test_statistical_functions_equivalence(self):
        k, n, p = 5, 10.0, 0.3
        scipy_nb = scipy.stats.nbinom.logpmf(k, n, p)
        numba_nb = nbinom_logpmf_numba(k, n, p)
        assert abs(scipy_nb - numba_nb) < 1e-10

        k, n, alpha, beta = 3, 10, 2.0, 3.0
        scipy_bb = scipy.stats.betabinom.logpmf(k, n, alpha, beta)
        numba_bb = betabinom_logpmf_numba(k, n, alpha, beta)
        assert abs(scipy_bb - numba_bb) < 1e-10

    def test_data_equivalence(self, test_data):
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = test_data

        rdr_orig, baf_orig = compute_emission_probability_nb_betabinom(
            base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
        )

        rdr_numba, baf_numba = compute_emissions_efficient(
            base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
        )

        # Handle -inf values separately
        finite_mask_rdr = np.isfinite(rdr_orig)
        finite_mask_baf = np.isfinite(baf_orig)

        # Check finite values are close
        np.testing.assert_allclose(
            rdr_orig[finite_mask_rdr],
            rdr_numba[finite_mask_rdr],
            rtol=1e-10,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            baf_orig[finite_mask_baf],
            baf_numba[finite_mask_baf],
            rtol=1e-10,
            atol=1e-10,
        )

        # Check -inf positions match
        assert np.array_equal(np.isinf(rdr_orig), np.isinf(rdr_numba))
        assert np.array_equal(np.isinf(baf_orig), np.isinf(baf_numba))

        
class TestBenchmarks:
    def test_performance_comparison(self, test_data):
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = test_data

        compute_emissions_efficient(
            base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
        )

        start_time = time.time()
        rdr_orig, baf_orig = compute_emissions_original(
            base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
        )
        original_time = time.time() - start_time

        start_time = time.time()
        rdr_numba, baf_numba = compute_emissions_efficient(
            base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X
        )
        numba_time = time.time() - start_time

        speedup = original_time / numba_time
        
        print(f"\nPerformance comparison:")
        print(f"Original implementation: {original_time:.4f}s")
        print(f"Numba implementation: {numba_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Assert significant speedup (at least 2x)
        assert speedup > 2.0, f"Expected speedup > 2x, got {speedup:.2f}x"


if __name__ == "__main__":
    # Run with: pytest test_emissions.py -v --benchmark-only
    # Or: pytest test_emissions.py -v -m "not slow"
    pytest.main([__file__, "-v"])
