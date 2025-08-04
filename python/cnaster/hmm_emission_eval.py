import numpy as np
from math import lgamma, log, exp, sqrt
import numba
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


@njit(nogil=True, cache=True, fastmath=False, error_model="numpy", parallel=True)
def compute_emissions_nb(
    base_nb_mean,
    log_mu,
    alphas,
    X,
    n_states,
    n_obs,
    n_spots,
):
    # TODO zeros? -np.inf
    log_emission_rdr = np.full((n_states, n_obs, n_spots), 0.0)

    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                if base_nb_mean[obs, s] > 0:
                    nb_mean = base_nb_mean[obs, s] * exp(log_mu[i, s])
                    nb_var = nb_mean + alphas[i, s] * nb_mean * nb_mean
                    nb_std = sqrt(nb_var)

                    n, p = convert_params_numba(nb_mean, nb_std)
                    log_emission_rdr[i, obs, s] = nbinom_logpmf_numba(
                        X[obs, 0, s], n, p
                    )

    return log_emission_rdr

@njit(nogil=True, cache=True, fastmath=False, error_model="numpy", parallel=True)
def compute_emissions_bb(
    total_bb_RD,
    p_binom,
    taus,
    X,
    n_states,
    n_obs,
    n_spots,
):
    # TODO zeros? -np.inf
    log_emission_baf = np.full((n_states, n_obs, n_spots), 0.0)

    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                if total_bb_RD[obs, s] > 0:
                    alpha = p_binom[i, s] * taus[i, s]
                    beta = (1.0 - p_binom[i, s]) * taus[i, s]

                    log_emission_baf[i, obs, s] = betabinom_logpmf_numba(
                        X[obs, 1, s], total_bb_RD[obs, s], alpha, beta
                    )

    return log_emission_baf


def compute_emissions(
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

    log_emission_rdr = compute_emissions_nb(
        base_nb_mean,
        log_mu,
        alphas,
        X,
        n_states,
        n_obs,
        n_spots,
    )
    log_emission_baf = compute_emissions_bb(
        total_bb_RD,
        p_binom,
        taus,
        X,
        n_states,
        n_obs,
        n_spots,
    )

    return log_emission_rdr, log_emission_baf