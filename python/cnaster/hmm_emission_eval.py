import numpy as np
from math import lgamma, log, sqrt
import numba
import logging
from numba import njit


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def convert_params_numba(mean, std):
    var = std * std
    p = mean / var
    n = mean * p / (1.0 - p)
    return n, p


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def nbinom_logpmf_numba(k, r, p):
    if p <= 0.0 or p >= 1.0 or r <= 0.0:
        return 0.0

    if k < 0:
        return 0.0

    log_coeff = lgamma(k + r) - lgamma(k + 1) - lgamma(r)
    return log_coeff + r * log(p) + k * log(1.0 - p)


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def betabinom_logpmf_numba(k, n, alpha, beta):
    if alpha <= 0.0 or beta <= 0.0 or n < 0 or k < 0 or k > n:
        return 0.0

    log_binom_coeff = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
    log_beta_denom = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

    return log_binom_coeff + log_beta_num - log_beta_denom


# error_model="numpy"
@njit(nogil=True, cache=False, fastmath=False, parallel=False)
def compute_emissions_nb(
    X,
    base_nb_mean,
    tumor_prop,
    log_mu,
    alphas,
    n_states,
    n_obs,
    n_spots,
):
    # NB Solves either (num_states, num_obs, num_spots) for num_spot = 1 or concatenate of num_clones along obs axis.
    log_emission_rdr = np.full((n_states, n_obs, n_spots), 0.0)

    # TODO
    assert log_mu.shape[1] == 1

    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                if base_nb_mean[obs, s] > 0:
                    nb_mean = base_nb_mean[obs, s] * (
                        tumor_prop[obs, s] * np.exp(log_mu[i, 0])
                        + 1.
                        - tumor_prop[obs, s]
                    )

                    nb_var = nb_mean + alphas[i, 0] * nb_mean * nb_mean
                    nb_std = sqrt(nb_var)

                    n, p = convert_params_numba(nb_mean, nb_std)
                    log_emission_rdr[i, obs, s] = nbinom_logpmf_numba(
                        X[obs, 0, s], n, p
                    )

    return log_emission_rdr


# error_model="numpy"
@njit(nogil=False, cache=False, fastmath=False, parallel=False)
def compute_emissions_bb(
    X,
    total_bb_RD,
    tumor_prop,
    mu_weighted_tumor_prop,
    p_binom,
    taus,
    n_states,
    n_obs,
    n_spots,
):
    # NB Solves either (num_states, num_obs, num_spots) for num_spot = 1 or concatenate of num_clones along obs axis.
    log_emission_baf = np.full((n_states, n_obs, n_spots), 0.0)

    # TODO
    assert p_binom.shape[1] == 1

    # TODO HACK
    assert np.all(mu_weighted_tumor_prop == 1.0)
    assert np.all(tumor_prop == 1.0)

    # NB no phasing
    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                if total_bb_RD[obs, s] > 0:
                    pA = (p_binom[i, 0] * mu_weighted_tumor_prop[obs, s] + 0.5 * (1.0 - tumor_prop[obs, s])) / (mu_weighted_tumor_prop[obs, s] + 1. - tumor_prop[obs, s])

                    alpha = pA * taus[i, 0]
                    beta = (1. - pA) * taus[i, 0]
                    
                    log_emission_baf[i, obs, s] = betabinom_logpmf_numba(
                        X[obs, 1, s], total_bb_RD[obs, s], alpha, beta
                    )

    return log_emission_baf


def compute_emissions(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop, mu_weighted_tumor_prop):
    n_obs, _, n_spots = X.shape
    n_states = log_mu.shape[0]

    X = np.ascontiguousarray(X, dtype=np.int32)
    base_nb_mean = np.ascontiguousarray(base_nb_mean, dtype=np.float64)
    log_mu = np.ascontiguousarray(log_mu, dtype=np.float64)
    alphas = np.ascontiguousarray(alphas, dtype=np.float64)
    total_bb_RD = np.ascontiguousarray(total_bb_RD, dtype=np.int32)
    p_binom = np.ascontiguousarray(p_binom, dtype=np.float64)
    taus = np.ascontiguousarray(taus, dtype=np.float64)
    
    tumor_prop = np.ascontiguousarray(tumor_prop, dtype=np.float64)
    mu_weighted_tumor_prop = np.ascontiguousarray(mu_weighted_tumor_prop, dtype=np.float64)
    
    logging.info(f"Computing emission probabilities for known normal baseline={np.any(base_nb_mean > 0)}.")
    
    # NB defaults to zero if normal baseline is not defined.
    log_emission_rdr = compute_emissions_nb(
        X,
        base_nb_mean,
        tumor_prop,
        log_mu,
        alphas,
        n_states,
        n_obs,
        n_spots,
    )

    log_emission_baf = compute_emissions_bb(
        X,
        total_bb_RD,
        tumor_prop,
        mu_weighted_tumor_prop,
        p_binom,
        taus,
        n_states,
        n_obs,
        n_spots,
    )

    return log_emission_rdr, log_emission_baf
