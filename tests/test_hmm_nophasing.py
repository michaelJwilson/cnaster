import pytest
import numpy as np
import scipy.stats
from cnaster.hmm_emission_eval import compute_emissions, nbinom_logpmf_numba, betabinom_logpmf_numba
from cnaster.deprecated.hmm_nophasing import compute_emission_probability_nb_betabinom


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


@pytest.fixture
def large_count_test_data():
    """Test data with much larger count values to stress-test performance"""
    np.random.seed(456)
    n_obs, n_spots, n_states = 4_494, 5, 10

    # Much larger base means for higher counts
    base_nb_mean = np.random.exponential(50.0, (n_obs, n_spots))

    # NB models dropout
    base_nb_mean[np.random.random((n_obs, n_spots)) < 0.1] = 0

    log_mu = np.random.normal(1.0, 0.5, (n_states, n_spots))  # Higher mean
    alphas = np.random.exponential(0.1, (n_states, n_spots))

    # Much larger total read depths
    total_bb_RD = np.random.poisson(500, (n_obs, n_spots))
    total_bb_RD[np.random.random((n_obs, n_spots)) < 0.1] = 0

    p_binom = np.random.beta(2, 2, (n_states, n_spots))
    taus = np.random.exponential(10, (n_states, n_spots))

    X = np.zeros((n_obs, 2, n_spots), dtype=int)
    # Generate much larger counts
    X[:, 0, :] = np.random.poisson(base_nb_mean * 20)  # ~1000x larger RDR counts
    X[:, 1, :] = np.random.binomial(total_bb_RD, 0.5)  # ~25x larger BAF counts

    return base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X


class TestEquivalence:
    @pytest.mark.parametrize(
        "k,n,p",
        [
            (5, 10.0, 0.3),
            (0, 1.0, 0.5),
            (100, 50.0, 0.1),
            (1, 1.0, 0.999),
            (10, 0.1, 0.001),
        ],
    )
    def test_nbinom_equivalence(self, k, n, p):
        """Test negative binomial log PMF equivalence"""
        scipy_nb = scipy.stats.nbinom.logpmf(k, n, p)
        numba_nb = nbinom_logpmf_numba(k, n, p)

        if np.isfinite(scipy_nb):
            assert abs(scipy_nb - numba_nb) < 1e-10
        else:
            assert not np.isfinite(numba_nb)

    @pytest.mark.parametrize(
        "k,n,alpha,beta",
        [
            (3, 10, 2.0, 3.0),
            (0, 5, 1.0, 1.0),
            (10, 10, 0.5, 0.5),
            (1, 100, 10.0, 5.0),
            (50, 100, 2.0, 2.0),
        ],
    )
    def test_betabinom_equivalence(self, k, n, alpha, beta):
        """Test beta-binomial log PMF equivalence"""
        scipy_bb = scipy.stats.betabinom.logpmf(k, n, alpha, beta)
        numba_bb = betabinom_logpmf_numba(k, n, alpha, beta)

        if np.isfinite(scipy_bb):
            assert abs(scipy_bb - numba_bb) < 1e-10
        else:
            assert not np.isfinite(numba_bb)

    def test_edge_cases_nbinom(self):
        """Test negative binomial edge cases"""
        # Invalid parameters should return 0.0
        assert nbinom_logpmf_numba(5, -1.0, 0.5) == 0.0  # negative n
        assert nbinom_logpmf_numba(5, 10.0, 0.0) == 0.0  # p = 0
        assert nbinom_logpmf_numba(5, 10.0, 1.0) == 0.0  # p = 1
        assert nbinom_logpmf_numba(-1, 10.0, 0.5) == 0.0  # negative k

    def test_edge_cases_betabinom(self):
        """Test beta-binomial edge cases"""
        # Invalid parameters should return 0.0
        assert betabinom_logpmf_numba(5, 10, -1.0, 2.0) == 0.0  # negative alpha
        assert betabinom_logpmf_numba(5, 10, 2.0, -1.0) == 0.0  # negative beta
        assert betabinom_logpmf_numba(-1, 10, 2.0, 2.0) == 0.0  # negative k
        assert betabinom_logpmf_numba(11, 10, 2.0, 2.0) == 0.0  # k > n
        assert betabinom_logpmf_numba(5, -1, 2.0, 2.0) == 0.0  # negative n

    def test_data_equivalence_comprehensive(self, test_data):
        """Comprehensive test of data equivalence"""
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = test_data

        rdr_orig, baf_orig = compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        rdr_numba, baf_numba = compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
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

    def test_zero_handling(self):
        """Test handling of zero values in input data"""
        n_obs, n_spots, n_states = 10, 2, 3

        # All zeros case
        base_nb_mean = np.zeros((n_obs, n_spots))
        log_mu = np.random.normal(0, 0.5, (n_states, n_spots))
        alphas = np.random.exponential(0.1, (n_states, n_spots))

        total_bb_RD = np.zeros((n_obs, n_spots), dtype=int)
        p_binom = np.random.beta(2, 2, (n_states, n_spots))
        taus = np.random.exponential(10, (n_states, n_spots))

        X = np.zeros((n_obs, 2, n_spots), dtype=int)

        rdr_orig, baf_orig = compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        rdr_numba, baf_numba = compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        np.testing.assert_allclose(rdr_orig, rdr_numba, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(baf_orig, baf_numba, rtol=1e-10, atol=1e-10)


class TestBenchmarks:
    def test_original_implementation_benchmark(self, benchmark, test_data):
        """Benchmark the original implementation"""
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = test_data

        result = benchmark(
            compute_emission_probability_nb_betabinom,
            X,
            base_nb_mean,
            log_mu,
            alphas,
            total_bb_RD,
            p_binom,
            taus,
        )

        assert len(result) == 2  # Should return (rdr, baf)

    def test_numba_implementation_benchmark(self, benchmark, test_data):
        """Benchmark the numba implementation"""
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = test_data

        # Warm up numba compilation
        compute_emissions(
            X[:10], base_nb_mean[:10], log_mu, alphas, total_bb_RD[:10], p_binom, taus
        )

        result = benchmark(
            compute_emissions,
            X,
            base_nb_mean,
            log_mu,
            alphas,
            total_bb_RD,
            p_binom,
            taus,
        )

        assert len(result) == 2

    def test_large_counts_numba_benchmark(self, benchmark, large_count_test_data):
        """Benchmark numba implementation with large count values"""
        base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, X = (
            large_count_test_data
        )

        # Warm up numba compilation
        compute_emissions(
            X[:10], base_nb_mean[:10], log_mu, alphas, total_bb_RD[:10], p_binom, taus
        )

        result = benchmark(
            compute_emissions,
            X,
            base_nb_mean,
            log_mu,
            alphas,
            total_bb_RD,
            p_binom,
            taus,
        )

        assert len(result) == 2


if __name__ == "__main__":
    # Run with: pytest test_hmm_nophasing.py -v --benchmark-only
    # Or: pytest test_hmm_nophasing.py -v --benchmark-skip
    # Or: pytest test_hmm_nophasing.py -v
    pytest.main([__file__, "-v"])
