import numpy as np
import scipy.special
import scipy.optimize
import time
import numpy as np
from math import lgamma, log, exp  # sqrt

# import numba
import pprint
from numba import njit

# from cnaster.hmm_update import (
# update_emission_params_bb_nophasing_uniqvalues_mix,
# update_emission_params_nb_nophasing_uniqvalues,
# update_emission_params_nb_nophasing_uniqvalues_mix,
# update_startprob_nophasing,
# update_transition_nophasing,
# )
from math import lgamma
from cnaster.hmm_utils import (
    # compute_posterior_obs,
    compute_posterior_transition_nophasing,
    # construct_unique_matrix,
    # convert_params_disp,
    # mylogsumexp,
    # np_sum_ax_squeeze,
    # get_em_solver_params,
)
from cnaster.count_encoder import CountEncoder

# from cnaster.hmm_emission_eval import compute_emissions

# from cnaster.hmm_emission import nloglikeobs_nb, nloglikeobs_bb

# from cnaster.hmm_sitewise import (
# compute_emission_probability_nb_betabinom_phased,
# forward_marginalize_phased,
# backward_marginalize_phased,
# )
from scipy.optimize import OptimizeResult
from numba import njit, prange
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

"""
@njit(nogil=True, cache=True, fastmath=False, error_model="numpy")
def convert_params_numba(mean, std):
    # NB negative binomial (n, p) given mean and std.
    # TODO better parameterization for numerical stability.
    var = std * std
    p = mean / var
    n = mean * p / (1.0 - p)
    return n, p

@njit
def convert_params_disp(mean, overdisp):
    p = 1.0 / (1.0 + overdisp * mean)

    # NB guard on min. overdispersion, such that (overdisp * mean) << 1.
    n = 1.0 / np.maximum(overdisp, 1.0e-10)

    return n, p
"""


@njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def nbinom_logpmf_numba(k, r, p):
    if p <= 0.0 or p >= 1.0 or r <= 0.0 or k < 0:
        return 0.0

    # TODO keyword to drop parameter-independent terms.
    log_coeff = lgamma(k + r) - lgamma(k + 1) - lgamma(r)
    return log_coeff + r * log(p) + k * log(1.0 - p)


@njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def betabinom_logpmf_numba(k, n, alpha, beta):
    if alpha <= 0.0 or beta <= 0.0 or n < 0 or k < 0 or k > n:
        return 0.0

    # TODO keyword to drop parameter-independent terms.
    log_binom_coeff = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
    log_beta_denom = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

    return log_binom_coeff + log_beta_num - log_beta_denom


"""
@njit(nogil=True, cache=True, fastmath=False, parallel=True, error_model="numpy")
def compute_emissions_nb(
    X,
    base_nb_mean,
    log_mu,
    alphas,
    n_states,
    n_obs,
    n_spots,
):
    # TODO guard against log_mu parameters defined with a "spot" (clone) axis > 1.
    assert log_mu.shape[1] == 1

    assert log_mu.shape[0] == n_states
    assert X.shape[0] == n_obs
    assert X.shape[2] == n_spots

    # TODO in-place scratch array.
    log_emission_rdr = np.full((n_states, n_obs, n_spots), 0.0)

    # TODO parallel actually faster?
    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                # TODO lift out?
                if base_nb_mean[obs, s] > 0:
                    nb_mean = base_nb_mean[obs, s] * exp(log_mu[i, 0])
                    nb_var = nb_mean + alphas[i, 0] * nb_mean**2.0
                    nb_std = sqrt(nb_var)

                    n, p = convert_params_numba(nb_mean, nb_std)
                    log_emission_rdr[i, obs, s] = nbinom_logpmf_numba(
                        X[obs, 0, s], n, p
                    )

    return log_emission_rdr


@njit(nogil=True, cache=True, fastmath=False, parallel=True, error_model="numpy")
def compute_emissions_bb(
    X,
    total_bb_RD,
    p_binom,
    taus,
    n_states,
    n_obs,
    n_spots,
):
    # TODO guard against p_binom parameters defined with a "spot" (clone) axis > 1.
    assert p_binom.shape[1] == 1

    assert p_binom.shape[0] == n_states
    assert X.shape[0] == n_obs
    assert X.shape[2] == n_spots

    # TODO in-place scratch array.
    log_emission_baf = np.full((n_states, n_obs, n_spots), 0.0)

    # TODO parallel actually faster?
    for i in numba.prange(n_states):
        for obs in range(n_obs):
            for s in range(n_spots):
                if total_bb_RD[obs, s] > 0:
                    # TODO lift out?
                    alpha = p_binom[i, 0] * taus[i, 0]
                    beta = (1.0 - p_binom[i, 0]) * taus[i, 0]

                    log_emission_baf[i, obs, s] = betabinom_logpmf_numba(
                        X[obs, 1, s], total_bb_RD[obs, s], alpha, beta
                    )

    return log_emission_baf


# TODO in-place scratch array.
def compute_emissions(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus):
    n_obs, _, n_spots = X.shape
    n_states = log_mu.shape[0]

    # DEPRECATE
    base_nb_mean = np.ascontiguousarray(base_nb_mean, dtype=np.float64)
    log_mu = np.ascontiguousarray(log_mu, dtype=np.float64)
    alphas = np.ascontiguousarray(alphas, dtype=np.float64)
    total_bb_RD = np.ascontiguousarray(total_bb_RD, dtype=np.int32)
    p_binom = np.ascontiguousarray(p_binom, dtype=np.float64)
    taus = np.ascontiguousarray(taus, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.int32)

    log_emission_rdr = compute_emissions_nb(
        X,
        base_nb_mean,
        log_mu,
        alphas,
        n_states,
        n_obs,
        n_spots,
    )
    log_emission_baf = compute_emissions_bb(
        X,
        total_bb_RD,
        p_binom,
        taus,
        n_states,
        n_obs,
        n_spots,
    )

    return log_emission_rdr, log_emission_baf
"""


@njit(nogil=True, cache=True, error_model="numpy")
def _nb_logpmf_1d(obs, exposure, mu, alpha):
    out = np.zeros_like(obs, dtype=np.float64)
    r = 1.0 / max(alpha, 1.0e-10)

    for i in range(len(obs)):
        k = obs[i]
        lambda_i = exposure[i] * mu

        if lambda_i <= 0.0:
            out[i] = 0.0
            continue

        p = 1.0 / (1.0 + alpha * lambda_i)
        out[i] = nbinom_logpmf_numba(k, r, p)

    return out


@njit(nogil=True, cache=True, error_model="numpy")
def _bb_logpmf_1d(obs, total, p_binom, tau, EPS=1e-10):
    out = np.zeros_like(obs, dtype=np.float64)
    alpha = max(p_binom * tau, EPS)
    beta = max((1.0 - p_binom) * tau, EPS)

    for i in range(len(obs)):
        out[i] = betabinom_logpmf_numba(obs[i], total[i], alpha, beta)

    return out


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_nb_logpmf(X_nb, base_nb_mean, log_mu, alphas):
    n_states = log_mu.shape[0]
    n_obs, n_spots = X_nb.shape

    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    for i in prange(n_states):
        mu_val = exp(log_mu[i, 0])
        alpha_val = alphas[i, 0]

        for s in range(n_spots):
            out[i, :, s] = _nb_logpmf_1d(
                X_nb[:, s], base_nb_mean[:, s], mu_val, alpha_val
            )

    return out


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_bb_logpmf(X_bb, total_bb_RD, p_binom, taus, EPS=1e-10):
    n_states = p_binom.shape[0]
    n_obs, n_spots = X_bb.shape

    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    for i in prange(n_states):
        p_val = p_binom[i, 0]
        tau_val = taus[i, 0]

        for s in range(n_spots):
            out[i, :, s] = _bb_logpmf_1d(
                X_bb[:, s], total_bb_RD[:, s], p_val, tau_val, EPS
            )

    return out


"""
@njit
def np_sum_ax_squeeze(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.zeros(arr.shape[1])

        for i in range(len(result)):
            result[i] = np.sum(arr[:, i])
    else:
        result = np.empty(arr.shape[0])

        for i in range(len(result)):
            result[i] = np.sum(arr[i, :])

    return result
"""


@njit
def np_sum_ax_squeeze(arr, axis=0):
    return np.sum(arr, axis=axis)


"""
@njit
def mylogsumexp(a):
    a_max = np.max(a)

    if np.isinf(a_max):
        return a_max

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    s = np.log(s)

    return s + a_max
"""


@njit
def numba_logsumexp(a):
    a_max = np.max(a)
    if np.isinf(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


"""
def nloglikeobs_nb(
    endog,
    exog,
    weights,
    exposure,
    params,
    reduce=True,
):
    num_states = exog.shape[-1]
    nb_mean = exog @ np.exp(params[:num_states]) * exposure
    nb_disp = exog @ params[num_states:]

    # NB vectorized call.
    n, p = convert_params_disp(nb_mean, nb_disp)

    result = -scipy.stats.nbinom.logpmf(endog, n, p)
    result[np.isnan(result)] = np.inf

    if reduce:
        result = result.dot(weights)
        assert not np.isnan(result), f"{params}: {result}"

    return result


@njit(nogil=True, cache=True, error_model="numpy")
def betabinom_logpmf(endog, exposure, a, b, zero_point, EPS=1.0e-10):
    result_array = np.empty_like(endog, dtype=np.float64)

    for i in range(len(endog)):
        ai = a[i]
        bi = b[i]

        # NB guard against numerical instability at 0
        if ai < EPS:
            ai = EPS
        if bi < EPS:
            bi = EPS

        result_array[i] = (
            zero_point[i]
            + lgamma(endog[i] + ai)
            + lgamma(exposure[i] - endog[i] + bi)
            + lgamma(ai + bi)
            - lgamma(exposure[i] + ai + bi)
            - lgamma(ai)
            - lgamma(bi)
        )
        if np.isnan(result_array[i]):
            result_array[i] = -np.inf

    return result_array


@njit(nogil=True, cache=True, error_model="numpy")
def compute_bb_ab(exog, params):
    num_states = exog.shape[-1]

    p = np.dot(exog, params[:num_states])
    t = np.dot(exog, params[num_states:])

    a = p * t
    b = (1.0 - p) * t

    return a, b


def nloglikeobs_bb(
    endog,
    exog,
    weights,
    exposure,
    params,
    zero_point=None,
    reduce=True,
):
    a, b = compute_bb_ab(exog, params)

    if zero_point is not None:
        result = -betabinom_logpmf(endog, exposure, a, b, zero_point)
    else:
        result = -scipy.stats.betabinom.logpmf(endog, exposure, a, b)
        result[np.isnan(result)] = np.inf

    if reduce:
        reduced_result = result.dot(weights)

        if np.isnan(reduced_result):
            logger.info(
                f"Detected invalid ln. likelihood={reduced_result} for:\n{params}"
            )

            nan_mask = np.isnan(weights)
            nan_weights = weights[nan_mask]

            nan_mask = np.isnan(result)
            nan_endog = np.unique(endog[nan_mask])
            nan_exposure = np.unique(exposure[nan_mask])
            nan_alphas = np.unique(a[nan_mask])
            nan_betas = np.unique(b[nan_mask])

            logger.info(
                f"NaN identified:\n"
                f"  weights: {nan_weights}\n"
                f"  endog: {nan_endog}\n"
                f"  exposure: {nan_exposure}\n"
                f"  alphas: {nan_alphas}\n"
                f"  betas: {nan_betas}\n"
                f"  Fraction of NaN observations: {np.mean(nan_mask):.6e}"
            )

            raise RuntimeError()

        result = reduced_result

    return result
"""


def get_log_transmat(n_states, t):
    if n_states > 1:
        transmat = np.ones((n_states, n_states)) * (1.0 - t) / (n_states - 1)
        np.fill_diagonal(transmat, t)
        log_transmat = np.log(transmat)
    else:
        log_transmat = np.zeros((1, 1))

    return log_transmat


class hmm_nophasing:
    def __init__(self, params="stmp", t=1 - 1e-4):
        self.params = params
        self.t = t

        # NB small alpha tend to Poisson. 0.1 ->
        # NB large dispersions tend to Binomial, flat landscape, initialize just before.  30 -> 1_000
        # self.n_states = n_states
        # self.default_log_mu = np.linspace(-0.1, 0.1, n_states)
        # self.default_p_binom = np.linspace(0.05, 0.45, n_states)
        # self.default_alphas = 0.5 * np.ones(n_states)
        # self.default_taus = 1_000 * np.ones(n_states)
        # self.default_log_startprob = np.log(np.ones(n_states) / n_states)

    """
    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        return compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    """

    # TODO call coded
    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        # X is shape (n_obs, 2, n_spots). Split into NB (index 0) and BB (index 1) arrays
        log_emit_rdr = _dense_nb_logpmf(X[:, 0, :], base_nb_mean, log_mu, alphas)
        log_emit_baf = _dense_bb_logpmf(X[:, 1, :], total_bb_RD, p_binom, taus)

        return log_emit_rdr, log_emit_baf

    """
    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        # TODO assumes called on each clone independently.
        n_states = log_mu.shape[0]

        # TODO guard against log_mu parameters defined with a "spot" (clone) axis > 1.
        assert log_mu.shape[1] == 1

        # NB assumes a single spot, index 0.
        nb_endog = nbEncoder.get_unique_obs(0)
        nb_exposure = nbEncoder.get_unique_total(0)
        nb_valid = nb_exposure > 0

        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)
        bb_valid = bb_exposure > 0

        # TODO in-place scratch array.
        nb_ones = np.ones_like(nb_endog, dtype=float).reshape(-1, 1)
        bb_ones = np.ones_like(bb_endog, dtype=float).reshape(-1, 1)

        # TODO in-place scratch array.
        log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
        log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

        for i in range(n_states):
            if np.any(nb_valid):
                log_emit_rdr_uniq[i, nb_valid] = -nloglikeobs_nb(
                    nb_endog[nb_valid],
                    nb_ones[nb_valid],
                    nb_ones[nb_valid],
                    nb_exposure[nb_valid],
                    np.array([log_mu[i, 0], alphas[i, 0]]),
                    reduce=False,
                )
            else:
                log_emit_rdr_uniq[:, :] = 0.0

            if np.any(bb_valid):
                log_emit_baf_uniq[i, bb_valid] = -nloglikeobs_bb(
                    bb_endog[bb_valid],
                    bb_ones[bb_valid],
                    bb_ones[bb_valid],
                    bb_exposure[bb_valid],
                    np.array([p_binom[i, 0], taus[i, 0]]),
                    reduce=False,
                )
            else:
                log_emit_baf_uniq[:, :] = 0.0

        # TODO zero?
        log_emit_rdr = nbEncoder.decode_array(log_emit_rdr_uniq, 0)
        log_emit_baf = bbEncoder.decode_array(log_emit_baf_uniq, 0)

        return log_emit_rdr, log_emit_baf
    """

    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        # TODO assumes called on each clone independently.
        n_states = log_mu.shape[0]

        nb_endog = nbEncoder.get_unique_obs(0)
        nb_exposure = nbEncoder.get_unique_total(0)

        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
        log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

        for i in range(n_states):
            log_emit_rdr_uniq[i, :] = _nb_logpmf_1d(
                nb_endog, nb_exposure, exp(log_mu[i, 0]), alphas[i, 0]
            )
            log_emit_baf_uniq[i, :] = _bb_logpmf_1d(
                bb_endog, bb_exposure, p_binom[i, 0], taus[i, 0]
            )

        log_emit_rdr = nbEncoder.decode_array(log_emit_rdr_uniq, 0)
        log_emit_baf = bbEncoder.decode_array(log_emit_baf_uniq, 0)

        return log_emit_rdr, log_emit_baf

    """
    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        n_states = log_mu.shape[0]
        n_spots = nbEncoder.n_spots

        assert bbEncoder.n_spots == n_spots
        
        log_emit_rdr_list, log_emit_baf_list = [],[]

        for s in range(n_spots):
            nb_endog = nbEncoder.get_unique_obs(s)
            nb_exposure = nbEncoder.get_unique_total(s)

            bb_endog = bbEncoder.get_unique_obs(s)
            bb_exposure = bbEncoder.get_unique_total(s)

            log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
            log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

            for i in range(n_states):
                log_emit_rdr_uniq[i, :] = _nb_logpmf_1d(
                    nb_endog, nb_exposure, exp(log_mu[i, s]), alphas[i, s]
                )
                log_emit_baf_uniq[i, :] = _bb_logpmf_1d(
                    bb_endog, bb_exposure, p_binom[i, s], taus[i, s]
                )

            log_emit_rdr_list.append(nbEncoder.decode_array(log_emit_rdr_uniq, s))
            log_emit_baf_list.append(bbEncoder.decode_array(log_emit_baf_uniq, s))

        log_emit_rdr = np.stack(log_emit_rdr_list, axis=2)
        log_emit_baf = np.stack(log_emit_baf_list, axis=2)

        return log_emit_rdr, log_emit_baf
    """
    """
    @staticmethod
    def compute_emission_probability_nb_betabinom_mix(
        X,
        base_nb_mean,
        log_mu,
        alphas,
        total_bb_RD,
        p_binom,
        taus,
        tumor_prop,
        **kwargs,
    ):
        n_obs, _, n_spots = X.shape
        n_states = log_mu.shape[0]

        log_emission_rdr = np.zeros((n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((n_states, n_obs, n_spots))

        for i in np.arange(n_states):
            for s in np.arange(n_spots):
                idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0)[0]

                if len(idx_nonzero_rdr) > 0:
                    nb_mean = base_nb_mean[idx_nonzero_rdr, s] * (
                        tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[i, s])
                        + 1
                        - tumor_prop[idx_nonzero_rdr, s]
                    )
                    n, p = convert_params_disp(nb_mean, alphas[i, s])
                    log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(
                        X[idx_nonzero_rdr, 0, s], n, p
                    )

                # TODO brittle.
                if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
                    this_weighted_tp = []

                    # TODO assumes clone (and contig?) stacked
                    for c in range(len(kwargs["sample_length"])):
                        range_s = np.sum(kwargs["sample_length"][:c])
                        range_t = np.sum(kwargs["sample_length"][: (c + 1)])

                        this_weighted_tp.append(
                            tumor_prop[range_s:range_t, s]
                            * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c, s])
                            / (
                                tumor_prop[range_s:range_t, s]
                                * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c, s])
                                + 1
                                - tumor_prop[range_s:range_t, s]
                            )
                        )

                    this_weighted_tp = np.concatenate(this_weighted_tp)
                else:
                    this_weighted_tp = tumor_prop[:, s]

                idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0)[0]

                if len(idx_nonzero_baf) > 0:
                    mix_p_A = p_binom[i, s] * this_weighted_tp[
                        idx_nonzero_baf
                    ] + 0.5 * (1.0 - this_weighted_tp[idx_nonzero_baf])

                    mix_p_B = (1.0 - p_binom[i, s]) * this_weighted_tp[
                        idx_nonzero_baf
                    ] + 0.5 * (1.0 - this_weighted_tp[idx_nonzero_baf])

                    log_emission_baf[
                        i, idx_nonzero_baf, s
                    ] += scipy.stats.betabinom.logpmf(
                        X[idx_nonzero_baf, 1, s],
                        total_bb_RD[idx_nonzero_baf, s],
                        mix_p_A * taus[i, s],
                        mix_p_B * taus[i, s],
                    )

        return log_emission_rdr, log_emission_baf
    """

    @staticmethod
    @njit
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        """
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.

        Input
            log_emission: n_states * n_observations * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states.  Transition probability.
            log_startprob: n_states. Start probability after log transformation.
        Output
            log_alpha: size n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]

        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"

        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"

        log_alpha = np.zeros((n_states, n_obs))
        buf = np.zeros(n_states)
        cumlen = 0

        for le in lengths:
            # NB initialize with start_prob and emission of first obs. for each item of lengths,
            #    e.g. contig.  Treats last axis (spots/clones) as iid (TBC).
            log_alpha[:, cumlen] = log_startprob + np_sum_ax_squeeze(
                log_emission[:, cumlen, :], axis=1
            )

            for t in np.arange(1, le):
                for j in np.arange(n_states):
                    for i in np.arange(n_states):
                        buf[i] = log_alpha[i, (cumlen + t - 1)] + log_transmat[i, j]

                    log_alpha[j, (cumlen + t)] = numba_logsumexp(buf) + np.sum(
                        log_emission[j, (cumlen + t), :]
                    )

            cumlen += le

        return log_alpha

    @staticmethod
    @njit
    def backward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        """
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.

        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: n_states * n_observations * n_spots. Log probability.
        Output
            log_beta: (n_states * n_observations). log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]
        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"
        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"

        log_beta = np.zeros((n_states, n_obs))
        buf = np.zeros(n_states)
        cumlen = 0
        for le in lengths:
            log_beta[:, (cumlen + le - 1)] = 0

            for t in np.arange(le - 2, -1, -1):
                for i in np.arange(n_states):
                    for j in np.arange(n_states):
                        buf[j] = (
                            log_beta[j, (cumlen + t + 1)]
                            + log_transmat[i, j]
                            + np.sum(log_emission[j, (cumlen + t + 1), :])
                        )
                    log_beta[i, (cumlen + t)] = numba_logsumexp(buf)
            cumlen += le
        return log_beta

    # TODO rename get_log_state_posteriors.
    def get_state_posteriors(
        self, lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        log_alpha = self.forward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        log_beta = self.backward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        # NB log_gamma (n_states * n_observations), potentially concatenated by clone.
        log_gamma = log_alpha + log_beta

        if np.any(np.sum(log_gamma, axis=0) == 0):
            logger.error("Sum of posterior probability is zero for some observations!")
            raise RuntimeError()

        # NB normalize across states for each observation.
        log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)

        return log_gamma

    # DEPRECATE
    def get_transition_posteriors(
        self, lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        log_alpha = self.forward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        log_beta = self.backward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        return compute_posterior_transition_nophasing(
            log_alpha,
            log_beta,
            log_transmat,
            log_emission,
        )

    # TODO define self.n_states
    def get_initial_params(
        self,
        n_states,
        n_spots,
        init_log_mu=None,  # DEPRECATE
        init_p_binom=None,  # DEPRECATE
        init_alphas=None,  # DEPRECATE
        init_taus=None,  # DEPRECATE
    ):
        # TODO use self.default_log_mu on class instance
        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for _ in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )

        # TODO define self.default_p_binom on class instance
        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for _ in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )

        # NB small alpha tend to Poisson. 0.1 ->
        alphas = (
            0.5 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )

        # TODO define ...
        # NB large dispersions tend to Binomial, flat landscape, initialize just before.  30 -> 1_000
        taus = 1_000 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        # NB initialize start probability and emission probability
        log_startprob = np.log(np.ones(n_states) / n_states)

        """
        # TODO definse self.trans_mat on class instance
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1.0 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))
        """
        log_transmat = get_log_transmat(n_states, self.t)

        return log_mu, p_binom, alphas, taus, log_startprob, log_transmat

    def get_bounds(
        self,
        n_states,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
        max_alpha=1_000.0,
        min_alpha=1e-6,
        max_tau=5_000.0,
        min_tau=1e-4,
    ):
        """
        Dynamically constructs the bounds list of (min, max) tuples to exactly
        match the flattened optimization vector generated by pack_params.
        """
        bounds = []

        if "s" in self.params:
            # Unconstrained because they pass through a Softmax upon unpack
            bounds.extend([(None, None)] * n_states)

        if optimize_nb and "m" in self.params:
            bounds.extend([(None, None)] * n_states)

        if "p" in self.params:
            if use_logit:
                # Logit constraint automatically bounds domain to (0, 1) upon unpack
                bounds.extend([(None, None)] * n_states)
            else:
                bounds.extend([(1e-6, 1.0 - 1e-6)] * n_states)

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            alpha_bnds = (float(np.log(min_alpha)), float(np.log(max_alpha)))
            if shared_NB_dispersion:
                bounds.append(alpha_bnds)
            else:
                bounds.extend([alpha_bnds] * n_states)

        if "p" in self.params and not fix_BB_dispersion:
            tau_bnds = (float(np.log(min_tau)), float(np.log(max_tau)))

            if shared_BB_dispersion:
                bounds.append(tau_bnds)
            else:
                bounds.extend([tau_bnds] * n_states)

        return bounds

    def pack_params(
        self,
        log_startprob,
        log_mu,
        p_binom,
        alphas,
        taus,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        """
        Defines an optimization vector given canonical parameterization & runtime settings.
        """
        # TODO parameter block, dispersion block, parameter block, dispersion block, etc.
        params_list = []
        if "s" in self.params:
            params_list.append(log_startprob.flatten())

        # TODO assert m not in self.params if optimize_nb is False on class instance,
        #      drop branch clause.
        if optimize_nb and "m" in self.params:
            params_list.append(log_mu.flatten())

        if "p" in self.params:
            params_list.append(
                scipy.special.logit(p_binom.flatten())
                if use_logit
                else p_binom.flatten()
            )

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                params_list.append(np.array([np.log(alphas.flatten()[0])]))
            else:
                params_list.append(np.log(alphas.flatten()))

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                params_list.append(np.array([np.log(taus.flatten()[0])]))
            else:
                params_list.append(np.log(taus.flatten()))

        return np.concatenate(params_list) if params_list else np.array([])

    def unpack_params(
        self,
        x,
        n_states,
        log_startprob_init,
        log_mu_init,
        p_binom_init,
        alphas_init,
        taus_init,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        """
        Reconstruct canonical parameterization given optimization vector & runtime settings.
        """
        idx = 0

        if "s" in self.params:
            raw_startprob = x[idx : idx + n_states]
            log_startprob = raw_startprob - scipy.special.logsumexp(raw_startprob)
            idx += n_states
        else:
            log_startprob = log_startprob_init

        # NB returns default log_mu if not optimizing NB mean.
        if optimize_nb and "m" in self.params:
            log_mu = x[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            log_mu = log_mu_init

        if "p" in self.params:
            if use_logit:
                p_binom = scipy.special.expit(
                    x[idx : idx + n_states].reshape(n_states, 1)
                )
            else:
                p_binom = np.clip(
                    x[idx : idx + n_states].reshape(n_states, 1), 1e-6, 1 - 1e-6
                )
            idx += n_states
        else:
            p_binom = p_binom_init

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                val = np.exp(x[idx])
                alphas = np.full((n_states, 1), val)
                idx += 1
            else:
                alphas = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            alphas = alphas_init

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                val = np.exp(x[idx])
                taus = np.full((n_states, 1), val)
                idx += 1
            else:
                taus = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            taus = taus_init

        return log_startprob, log_mu, p_binom, alphas, taus

    # WARNING
    def unpack_param_errors(
        self,
        x,
        hess_inv,
        n_states,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        idx = 0
        parameter_errors_diag = np.sqrt(np.clip(np.diag(hess_inv), a_min=0, a_max=None))

        if "s" in self.params:
            raw_startprob = x[idx : idx + n_states]
            cov_raw = hess_inv[idx : idx + n_states, idx : idx + n_states]

            p_start = scipy.special.softmax(raw_startprob)

            J = np.eye(n_states) - np.outer(np.ones(n_states), p_start)

            cov_transformed = J @ cov_raw @ J.T

            log_startprob_err = np.sqrt(
                np.clip(np.diag(cov_transformed), a_min=0, a_max=None)
            )
            idx += n_states
        else:
            log_startprob_err = None

        if optimize_nb and "m" in self.params:
            log_mu_err = parameter_errors_diag[idx : idx + n_states].reshape(
                n_states, 1
            )
            idx += n_states
        else:
            log_mu_err = None

        if "p" in self.params:
            raw_p_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
            if use_logit:
                p_binom_val = scipy.special.expit(
                    x[idx : idx + n_states].reshape(n_states, 1)
                )
                p_binom_err = p_binom_val * (1 - p_binom_val) * raw_p_err
            else:
                p_binom_err = raw_p_err
            idx += n_states
        else:
            p_binom_err = None

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]

                alpha_val = np.exp(val_raw)
                alpha_err = alpha_val * val_err
                alphas_err = np.full((n_states, 1), alpha_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )

                alphas_val = np.exp(val_raw)
                alphas_err = alphas_val * val_err
                idx += n_states
        else:
            alphas_err = None

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]

                tau_val = np.exp(val_raw)
                tau_err = tau_val * val_err
                taus_err = np.full((n_states, 1), tau_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )

                taus_val = np.exp(val_raw)
                taus_err = taus_val * val_err
                idx += n_states
        else:
            taus_err = None

        return log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err

    '''
    def run_baum_welch_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat=None,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=100,
        tol=1e-4,
        **kwargs,
    ):
        """
        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            base_nb_mean: size of n_observations * n_spots.
            In NB-BetaBinom model, n_components = 2
        Intermediate
            log_mu: size of n_states. Log of mean/exposure/base_prob of each HMM state.
            alpha: size of n_states. Dispersioon parameter of each HMM state.
        """
        _, n_comp, n_spots = X.shape

        # NB TODO code treats spot axis as iid emission.
        #         expects clones to be concatenated along obs. axis or passed separately.
        #         also true of the "mapping" compression for unique emission configurations.
        assert n_spots == 1
        assert n_comp == 2

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.get_initial_params(
            n_states,
            n_spots,
            init_log_mu,
            init_p_binom,
            init_alphas,
            init_taus,
        )

        kwargs_str = (
            "{\n" + "\n".join(f"  '{k}': {v}" for k, v in kwargs.items()) + "\n}"
            if kwargs
            else "{}"
        )
        logger.info(f"Assuming kwargs={kwargs_str}")
        logger.info(
            f"Assumed initial p_binom and dispersion:\n{np.hstack((p_binom, taus))}"
        )

        # DEPRECATE utilize a pre-existing state posterior.
        log_gamma = kwargs.get("log_gamma", None)

        logger.info(f"Assumed initial log_gamma?  {log_gamma is not None}")

        # NB unique_values is a list of length "n_spots", read clones, each element is an array of
        #    shape (n_unique_pairs, 2) with columns of rounded (obs_count, total_count).
        #
        #    mapping_matrices is a list of length n_spots, read clones, each element is a sparse matrix
        #    of shape (n_obs, n_unique_pairs) mapping obs. to compressed space per spot.
        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(
            X[:, 0, :], base_nb_mean
        )

        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(
            X[:, 1, :], total_bb_RD
        )

        logger.info(
            "Constructed BB/NB compression in (X[:, 1, :], total_bb_RD) and (X[:, 0, :], base_nb_mean)."
        )

        for r in range(max_iter):
            logger.info(
                f"----  Solving for Baum-Welch iteration {r}/{max_iter} with Negative Binomial & Beta Binomial emission  -----"
            )

            if tumor_prop is None:
                # NB does not utilize unique_values.
                (
                    log_emission_rdr,
                    log_emission_baf,
                ) = self.compute_emission_probability_nb_betabinom(
                    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                )
            else:
                # NB estimates the spot library size in the absence of CNAs by scaling copy state log_mu;
                #    the correction is clone specific.
                if ((log_gamma is not None) or (r > 0)) and ("m" in self.params):
                    logmu_shift = []

                    for c in range(len(kwargs["sample_length"])):
                        this_pred_cnv = (
                            np.argmax(
                                log_gamma[
                                    :,
                                    np.sum(kwargs["sample_length"][:c]) : np.sum(
                                        kwargs["sample_length"][: (c + 1)]
                                    ),
                                ],
                                axis=0,
                            )
                            % n_states
                        )

                        logmu_shift.append(
                            scipy.special.logsumexp(
                                log_mu[this_pred_cnv, :]
                                + np.log(kwargs["lambd"]).reshape(-1, 1),
                                axis=0,
                            )
                        )

                    logmu_shift = np.vstack(logmu_shift)

                    logger.info(
                        f"Applying logmu_shift with median={np.median(logmu_shift)} and max={logmu_shift.max()}"
                    )

                    (
                        log_emission_rdr,
                        log_emission_baf,
                    ) = self.compute_emission_probability_nb_betabinom_mix(
                        X,
                        base_nb_mean,
                        log_mu,
                        alphas,
                        total_bb_RD,
                        p_binom,
                        taus,
                        tumor_prop,
                        logmu_shift=logmu_shift,
                        sample_length=kwargs["sample_length"],
                    )
                else:
                    (
                        log_emission_rdr,
                        log_emission_baf,
                    ) = self.compute_emission_probability_nb_betabinom_mix(
                        X,
                        base_nb_mean,
                        log_mu,
                        alphas,
                        total_bb_RD,
                        p_binom,
                        taus,
                        tumor_prop,
                    )

            log_emission = log_emission_rdr + log_emission_baf

            # NB e-step ... log_gamma (n_states * n_observations), potentially concatenated by clone.
            log_gamma = self.get_state_posteriors(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            contracted_log_gamma = np.sum(np.exp(log_gamma), axis=1) / np.sum(
                np.exp(log_gamma)
            )

            logger.info(
                f"State posterior breakdown:\n{[f'{xx:.4e}' for xx in contracted_log_gamma]}"
            )

            # HACK MAGIC TODO
            if contracted_log_gamma.min() < 1.0e-6:
                logger.warning(f"Defunct copy number states detected.")

            # NB m-step
            if "s" in self.params:
                new_log_startprob = update_startprob_nophasing(lengths, log_gamma)
                new_log_startprob = new_log_startprob.flatten()

                logger.info(
                    f"Updated HMM start probability=\n{[xx for xx in new_log_startprob]}"
                )
            else:
                new_log_startprob = log_startprob

            if "t" in self.params:
                log_xi = self.get_transition_posteriors(
                    lengths,
                    log_transmat,
                    log_startprob,
                    log_emission,
                    log_sitewise_transmat,
                )
                new_log_transmat = update_transition_nophasing(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat

            if "m" in self.params:
                if tumor_prop is None:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_nophasing_uniqvalues(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        alphas,
                        start_log_mu=log_mu,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                    )
                else:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_nophasing_uniqvalues_mix(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        alphas,
                        tumor_prop,
                        start_log_mu=log_mu,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                    )
            else:
                new_log_mu = log_mu
                new_alphas = alphas

            if "p" in self.params:
                if tumor_prop is None:
                    (
                        new_p_binom,
                        new_taus,
                    ) = update_emission_params_bb_nophasing_uniqvalues_mix(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        taus,
                        tumor_prop=None,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
                else:
                    # NB estimates the spot library size in the absence of CNAs by scaling copy state log_mu;
                    #    the correction is clone specific.
                    if "m" in self.params:
                        mu = []
                        for c in range(len(kwargs["sample_length"])):
                            this_pred_cnv = (
                                np.argmax(
                                    log_gamma[
                                        :,
                                        np.sum(kwargs["sample_length"][:c]) : np.sum(
                                            kwargs["sample_length"][: (c + 1)]
                                        ),
                                    ],
                                    axis=0,
                                )
                                % n_states
                            )
                            mu.append(
                                np.exp(new_log_mu[this_pred_cnv, :])
                                / np.sum(
                                    np.exp(new_log_mu[this_pred_cnv, :])
                                    * kwargs["lambd"].reshape(-1, 1),
                                    axis=0,
                                    keepdims=True,
                                )
                            )
                        mu = np.vstack(mu)
                        weighted_tp = (tumor_prop * mu) / (
                            tumor_prop * mu + 1 - tumor_prop
                        )
                    else:
                        weighted_tp = tumor_prop
                    (
                        new_p_binom,
                        new_taus,
                    ) = update_emission_params_bb_nophasing_uniqvalues_mix(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        taus,
                        weighted_tp,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
            else:
                new_p_binom = p_binom
                new_taus = taus

            logger.info(
                "Found max HMM parameter updates for tol=%.6e: \nstart prob.=%.6e\ntransfer matrix=%.6e\nmu=%.6e\np_binom=%.6e\nalpha=%.6e\ntau=%.6e",
                tol,
                np.max(np.abs(np.exp(new_log_startprob) - np.exp(log_startprob))),
                np.max(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))),
                np.max(np.abs(np.exp(new_log_mu) - np.exp(log_mu))),
                np.max(np.abs(new_p_binom - p_binom)),
                np.max(np.abs(new_alphas - alphas)),
                np.max(np.abs(new_taus - taus)),
            )

            # Warn if dispersion parameters increased
            if np.any(new_alphas > alphas):
                logger.warning(
                    f"NB dispersion (alpha) increased: max change = {np.max(new_alphas - alphas):.6e}"
                )
            if np.any(new_taus < taus):
                logger.warning(
                    f"BB dispersion (1/tau) increased (tau decreased): max change = {np.min(new_taus - taus):.6e}"
                )

            # NB log mu -> mu convergence.
            # TODO BUG? no check on start prob.
            transmat_converged = (
                np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))) < tol
            )

            # TODO HACK? np.exp(new_log_mu)
            log_mu_converged = (
                np.mean(np.abs(np.exp(new_log_mu) - np.exp(log_mu))) < tol
            )

            # TODO
            # mu_stds = np.sqrt(np.exp(new_log_mu) + alphas * np.exp(new_log_mu)**2)
            # log_mu_converged = (
            #    np.all(np.abs(np.exp(new_log_mu) - np.exp(log_mu)) < mu_stds / 5.)
            # )

            p_binom_converged = np.mean(np.abs(new_p_binom - p_binom)) < tol

            if transmat_converged and log_mu_converged and p_binom_converged:
                break
            else:
                logger.info(
                    f"Convergence of T, mu and p: {transmat_converged},{log_mu_converged},{p_binom_converged}"
                )

            log_startprob = new_log_startprob
            log_transmat = new_log_transmat
            log_mu = new_log_mu
            alphas = new_alphas
            p_binom = new_p_binom
            taus = new_taus
        else:
            logger.warning(f"hmm_nophasing failed to converge.")

        return {
            "new_log_mu": new_log_mu,
            "new_alphas": new_alphas,
            "new_p_binom": new_p_binom,
            "new_taus": new_taus,
            "new_log_startprob": new_log_startprob,  # TODO
            "new_log_transmat": new_log_transmat,  # TODO
            "log_gamma": log_gamma,
        }
    '''

    def optimize(self, *args, **kwargs):
        return self.run_baum_welch_nb_bb(*args, **kwargs)

    def run_baum_welch_nb_bb(self, *args, **kwargs):
        return self._run_optimization_pipeline("em", *args, **kwargs)

    def run_marg_likelihood_nb_bb(self, *args, **kwargs):
        return self._run_optimization_pipeline("marginal", *args, **kwargs)

    def _run_optimization_pipeline(
        self,
        mode,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat=None,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=1000,
        max_rdr=5.0,
        tol=1e-4,
        use_logit=True,
        propagate_errors=False,
        optimizer=None,
        **kwargs,
    ):
        _, n_comp, n_spots = X.shape
        assert (
            n_spots == 1
        ), "Currently expects (a) clone(s) concatenated along the genomic axis."
        assert n_comp == 2

        base_nb_mean = base_nb_mean.copy()
        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0
                base_nb_mean[est_rdr > max_rdr] = 0.0

        optimize_nb = np.any(base_nb_mean > 0)
        if "m" in self.params:
            assert (
                optimize_nb
            ), "Cannot optimize negative binomial if normal baseline is not defined."

        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

        logger.info(
            f"Encoders built. Medians: NB={np.median(nbEncoder.total_count):.4f} ({nbEncoder.compression_rate:.2%} comp), "
            f"BB={np.median(bbEncoder.total_count):.4f} ({bbEncoder.compression_rate:.2%} comp)."
        )

        (log_mu, p_binom, alphas, taus, log_startprob, log_transmat) = (
            self.get_initial_params(
                n_states, n_spots, init_log_mu, init_p_binom, init_alphas, init_taus
            )
        )

        kwargs_str = pprint.pformat(kwargs, indent=2) if kwargs else "{}"
        logger.info(
            f"--- hmm initialized ({mode.upper()}) ---\n"
            f"kwargs:\n{kwargs_str}\n"
            f"log_mu:\n{np.array2string(log_mu, precision=4, suppress_small=True)}\n"
            f"p_binom:\n{np.array2string(p_binom, precision=4, suppress_small=True)}\n"
            f"alphas:\n{np.array2string(alphas, precision=4, suppress_small=True)}\n"
            f"taus:\n{np.array2string(taus, precision=2, suppress_small=True)}\n"
            "--------------------------------"
        )

        x0 = self.pack_params(
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        if mode == "em":
            self.log_emissions, self.state_posteriors = None, None
            self.log_startprob = log_startprob
            self.iterations = 0
            default_optimizer = "BFGS"

            def callback(intermediate_result: OptimizeResult = None):
                if (self.iterations > 0) and (self.iterations % 2 != 0):
                    self.iterations += 1
                    return
                self.state_posteriors = np.exp(
                    self.get_state_posteriors(
                        lengths,
                        log_transmat,
                        self.log_startprob,
                        self.log_emissions,
                        log_sitewise_transmat,
                    )
                )
                self.iterations += 1

            def cost_fn(params):
                _, this_log_mu, this_p_binom, this_alphas, this_taus = (
                    self.unpack_params(
                        params,
                        n_states,
                        log_startprob,
                        log_mu,
                        p_binom,
                        alphas,
                        taus,
                        optimize_nb=optimize_nb,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                        use_logit=use_logit,
                    )
                )
                log_emission_rdr, log_emission_baf = (
                    self.compute_emission_probability_nb_betabinom_coded(
                        nbEncoder,
                        bbEncoder,
                        this_log_mu,
                        this_alphas,
                        this_p_binom,
                        this_taus,
                    )
                )
                self.log_emissions = (log_emission_rdr + log_emission_baf)[
                    :, :, np.newaxis
                ]

                if self.state_posteriors is None:
                    callback()
                return -np.sum(self.state_posteriors * self.log_emissions[..., 0])

        elif mode == "marginal":
            callback = None
            default_optimizer = "L-BFGS-B"

            def cost_fn(params):
                (
                    this_log_startprob,
                    this_log_mu,
                    this_p_binom,
                    this_alphas,
                    this_taus,
                ) = self.unpack_params(
                    params,
                    n_states,
                    log_startprob,
                    log_mu,
                    p_binom,
                    alphas,
                    taus,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
                log_emission_rdr, log_emission_baf = (
                    self.compute_emission_probability_nb_betabinom_coded(
                        nbEncoder,
                        bbEncoder,
                        this_log_mu,
                        this_alphas,
                        this_p_binom,
                        this_taus,
                    )
                )
                log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]
                log_alpha = self.forward_lattice(
                    lengths,
                    log_transmat,
                    this_log_startprob,
                    log_emissions,
                    log_sitewise_transmat,
                )

                curr, total_nll = 0, 0
                for le in lengths:
                    total_nll += -numba_logsumexp(log_alpha[:, curr + le - 1])
                    curr += le
                return total_nll

        else:
            raise ValueError(f"Unknown optimization mode: {mode}")

        opt_method = optimizer or default_optimizer
        options = {
            "maxiter": max_iter,
            "ftol": 1e-6,
            "gtol": 1e-5,
            "disp": False,
        } | kwargs.get("options", {})
        bounds = self.get_bounds(
            n_states,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        start_time_opt = time.time()
        logger.info(
            f"Starting {mode} optimization with {opt_method}. Initial cost={cost_fn(x0):.6e}"
        )

        res = scipy.optimize.minimize(
            cost_fn,
            x0,
            method=opt_method,
            bounds=bounds if mode == "marginal" else None,
            callback=callback,
            options=options,
        )

        logger.info(
            f"Optimization complete: {time.time() - start_time_opt:.2f}s | "
            f"{res.nit} iter | converged: {res.success} | NLL: {res.fun:.6e}"
        )

        final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus = (
            self.unpack_params(
                res.x,
                n_states,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
        )

        if propagate_errors:
            _, log_mu_err, p_binom_err, alphas_err, taus_err = self.unpack_param_errors(
                x=res.x,
                hess_inv=res.hess_inv,
                n_states=n_states,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
            param_errors = {
                "new_log_mu_err": log_mu_err,
                "new_alphas_err": alphas_err,
                "new_p_binom_err": p_binom_err,
                "new_taus_err": taus_err,
                "new_log_startprob_err": None,
            }
        else:
            param_errors = {}

        log_emission_rdr, log_emission_baf = (
            self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                final_log_mu,
                final_alphas,
                total_bb_RD,
                final_p_binom,
                final_taus,
            )
        )
        log_emission = log_emission_rdr + log_emission_baf
        log_gamma = self.get_state_posteriors(
            lengths,
            log_transmat,
            final_log_startprob,
            log_emission,
            log_sitewise_transmat,
        )
        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))

        log_lines = [
            f"--- Final HMM State ({self.__class__.__name__}) ---",
            f"p_binom:\n{np.array2string(final_p_binom, precision=3, suppress_small=True)}",
            f"taus:\n{np.array2string(final_taus, formatter={'float_kind': lambda x: f'{x:.3e}'})}",
        ]
        if optimize_nb:
            log_lines.extend(
                [
                    f"log_mu:\n{np.array2string(final_log_mu, precision=3, suppress_small=True)}",
                    f"alphas:\n{np.array2string(final_alphas, precision=3, suppress_small=True)}",
                ]
            )
        log_lines.extend(
            [
                f"State posteriors:\n{np.array2string(state_prior, formatter={'float_kind': lambda x: f'{x:.4e}'})}",
                f"Max updates (tol={tol:.6e}): mu={np.max(np.abs(np.exp(final_log_mu) - np.exp(log_mu))):.6e}",
            ]
        )
        logger.info("\n".join(log_lines))

        return {
            "new_log_mu": final_log_mu,
            "new_alphas": final_alphas,
            "new_p_binom": final_p_binom,
            "new_taus": final_taus,
            "new_log_startprob": final_log_startprob,
            "new_log_transmat": log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -res.fun,
            "n_states": n_states,
        } | param_errors

    '''
    def __run_baum_welch_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat=None,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,  # DEPRECATE
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=100,
        max_rdr=5.0,  # TODO HACK MAGIC
        tol=1e-4,
        use_logit=False,
        propagate_errors=False,
        optimizer="BFGS",
        **kwargs,
    ):
        _, _, n_spots = X.shape

        # NB currently, we expect to proceed clones concatenated along the genomic axis.
        assert n_spots == 1

        base_nb_mean = base_nb_mean.copy()
        optimize_nb = np.any(base_nb_mean > 0)

        if  "m" in self.params:
            assert optimize_nb, "Cannot optimize negative binomial if normal baseline is not defined."

        if optimize_nb and (max_rdr is not None):
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0

                # NB a local copy.
                base_nb_mean[est_rdr > max_rdr] = 0.0

        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :],  total_bb_RD)

        # NB solved for med. 26.0 and max. 75_849.0 total counts for bbEncoder.
        logger.info(
            f"Solved for med. {np.median(nbEncoder.total_count):.4f} and max. {np.max(nbEncoder.total_count):.4f} total counts for nbEncoder ({nbEncoder.compression_rate} compression rate)."
        )
        logger.info(
            f"Solved for med. {np.median(bbEncoder.total_count):.4f} and max. {np.max(bbEncoder.total_count):.4f} total counts for bbEncoder ({bbEncoder.compression_rate} compression rate)."
        )

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.get_initial_params(
            n_states,
            n_spots,
            init_log_mu, # None
            init_p_binom, # None
            init_alphas, # None
            init_taus, # None
        )

        # DEPRECATE utilize state posterior if given.
        log_gamma = kwargs.get("log_gamma", None)

        # TODO HACK?
        # init_alphas = 1. / np.exp(init_log_mu) if init_log_mu is not None else init_alphas
        # init_taus = np.median(bbEncoder.total_count) * np.ones_like(p_binom)

        # kwargs_str = (
        #     "{\n" + "\n".join(f"  '{k}': {v}" for k, v in kwargs.items()) + "\n}"
        #     if kwargs
        #     else "{}"
        # )

        kwargs_str = pprint.pformat(kwargs, indent=2) if kwargs else "{}"

        logger.info(
            "--- hmm initialized ---\n"
            f"kwargs:\n{kwargs_str}\n"
            f"log_mu:\n{np.array2string(log_mu, precision=4, suppress_small=True)}\n"
            f"p_binom:\n{np.array2string(p_binom, precision=4, suppress_small=True)}\n"
            f"alphas:\n{np.array2string(alphas, precision=4, suppress_small=True)}\n"
            f"taus:\n{np.array2string(taus, precision=2, suppress_small=True)}\n"
            "--------------------------------"
        )

        # NB pack parameters into a structure understood by scipy.optimize.minimize.
        x0 = self.pack_params(
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        self.log_emissions, self.state_posteriors = None, None
        self.log_startprob = log_startprob
        self.iterations = 0

        # TODO cadence of callback?
        # NB update state posteriors on (every other) scipy.optimize.minimize callback.
        def update_state_posteriors(intermediate_result: OptimizeResult = None):
            # TODO m-step for log_startprob and log_transmat.
            if (self.iterations > 0) and (self.iterations % 2 != 0):
                self.iterations += 1
                return

            # TODO rename get_log_state_posteriors.
            self.state_posteriors = np.exp(
                self.get_state_posteriors(
                    lengths,
                    log_transmat,
                    self.log_startprob,
                    self.log_emissions,
                    log_sitewise_transmat,
                )
            )

            self.iterations += 1

        def baum_welch_forward(params):
            # TODO log_startprob?
            _, this_log_mu, this_p_binom, this_alphas, this_taus = self.unpack_params(
                params,
                n_states,
                log_startprob,
                log_mu,  # TODO rename init_log_mu
                p_binom,
                alphas,  # TODO rename init_alphas
                taus,  # TODO rename init_taus
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )

            # NB emission is (nstates, n_observations, n_spots), but currently only supports n_spots=1.
            log_emission_rdr, log_emission_baf = (
                self.compute_emission_probability_nb_betabinom_coded(
                    nbEncoder,
                    bbEncoder,
                    this_log_mu,
                    this_alphas,
                    this_p_binom,
                    this_taus,
                )
            )
            """
            log_emission_rdr, log_emission_baf = self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                this_log_mu,
                this_alphas,
                total_bb_RD,
                this_p_binom,
                this_taus,
            )
            """

            # TBC add a clonal axis.  assumes a single "clone", potentially concatenated along the genomic axis.
            self.log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]
            # self.log_emissions = (log_emission_rdr + log_emission_baf)

            # NB log_gamma is (n_states * n_observations), potentially concatenated by clone on obs. axis.
            #    utilized on optimization callback.

            if self.state_posteriors is None:
                update_state_posteriors()

            # NB em cost is sum_iid of obs., sum_state of gamma * log_emission, which is negative log likelihood.
            return -np.sum(self.state_posteriors * self.log_emissions[..., 0])

        # NB vanilla max. likelihood or baum welch.
        cost, callback = baum_welch_forward, update_state_posteriors

        start_time_opt = time.time()
        logger.info(
            f"run_baum_welch_nb_bb with {optimizer}\nn_states={n_states};\nX.shape={X.shape};\nfixed_dispersion={fix_NB_dispersion};\nshared dispersion={shared_NB_dispersion};\noptimize_nb={optimize_nb};\nuse_logit={use_logit};\ninitial cost={cost(x0):.6e}"
        )

        options = {
            "maxiter": kwargs.get("max_iter", 10_000), # max_iter
            # "maxfun": kwargs.get("max_fun", 5_000),
            # "gtol": 1e-6,
            # "ftol": 1e-6,
            "disp": False,
        }

        bounds = self.get_bounds(
            n_states,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )
        
        res = scipy.optimize.minimize(
            cost,
            x0,
            method=optimizer,
            bounds=None, # use bounds.
            callback=callback,
            options=options,
        )

        end_time_opt = time.time()
        runtime = end_time_opt - start_time_opt

        logger.info(
            f"run_baum_welch_nb_bb complete: {runtime:.2f}s\n"
            f"{len(x0)} params,\n"
            f"{res.nit} iter,\n"
            f"{res.nfev} fcalls,\n"
            f"converged: {res.success},\n"
            f"message: {res.message},\n"
            f"nll: {res.fun:.6e}\n"
        )

        final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus = (
            self.unpack_params(
                res.x,
                n_states,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
        )

        if propagate_errors:
            # WARNING
            (_, log_mu_err, p_binom_err, alphas_err, taus_err) = (
                self.unpack_param_errors(
                    x=res.x,
                    hess_inv=res.hess_inv,
                    n_states=n_states,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
            )

            param_errors = {
                "new_log_mu_err": log_mu_err,
                "new_alphas_err": alphas_err,
                "new_p_binom_err": p_binom_err,
                "new_taus_err": taus_err,
                "new_log_startprob_err": None,  # TODO
            }
        else:
            param_errors = {}


        log_emission_rdr, log_emission_baf = (
            self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                final_log_mu,
                final_alphas,
                total_bb_RD,
                final_p_binom,
                final_taus,
            )
        )

        log_emission = log_emission_rdr + log_emission_baf

        log_gamma = self.get_state_posteriors(
            lengths,
            log_transmat,
            final_log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))

        """
        to_log = [
            f"Solved for best emission parameters with {self.__class__.__name__}:"
        ]

        to_log.append(f"p_binom=\n{[f'{xx:.3f}' for xx in final_p_binom[:,0]]}")
        to_log.append(f"taus=\n{[f'{xx:.3e}' for xx in final_taus[:,0]]}")

        if optimize_nb:
            to_log.append(f"mu=\n{[f'{xx:.3f}' for xx in final_log_mu[:,0]]}")
            to_log.append(f"alphas=\n{[f'{xx:.3f}' for xx in final_alphas[:,0]]}")

        logger.info("\n".join(to_log))

        logger.info(
            f"State posterior breakdown:\n{[f'{xx:.4e}' for xx in state_prior]}"
        )

        logger.info(
            "Found max HMM parameter updates for tol=%.6e: \nstart prob.=%.6e\ntransfer matrix=%.6e\nmu=%.6e\np_binom=%.6e\nalpha=%.6e\ntau=%.6e",
            tol,
            np.max(np.abs(np.exp(final_log_startprob) - np.exp(log_startprob))),
            np.max(np.abs(np.exp(log_transmat) - np.exp(log_transmat))),
            np.max(np.abs(np.exp(final_log_mu) - np.exp(log_mu))),
            np.max(np.abs(final_p_binom - p_binom)),
            np.max(np.abs(final_alphas - alphas)),
            np.max(np.abs(final_taus - taus)),
        )
        """

        log_lines = [
            f"--- Final HMM State ({self.__class__.__name__}) ---",
            f"p_binom:\n{np.array2string(final_p_binom, precision=3, suppress_small=True)}",
            f"taus:\n{np.array2string(final_taus, formatter={'float_kind': lambda x: f'{x:.3e}'})}"
        ]

        if optimize_nb:
            log_lines.extend([
                f"log_mu:\n{np.array2string(final_log_mu, precision=3, suppress_small=True)}",
                f"alphas:\n{np.array2string(final_alphas, precision=3, suppress_small=True)}"
            ])

        log_lines.extend([
            f"State posterior breakdown:\n{np.array2string(state_prior, formatter={'float_kind': lambda x: f'{x:.4e}'})}",
            "",
            f"Max parameter updates (tol={tol:.6e}):",
            f"  start prob. = {np.max(np.abs(np.exp(final_log_startprob) - np.exp(log_startprob))):.6e}",
            f"  trans. mat. = {np.max(np.abs(np.exp(log_transmat) - np.exp(log_transmat))):.6e}", # TODO BUG?
            f"  mu          = {np.max(np.abs(np.exp(final_log_mu) - np.exp(log_mu))):.6e}",
            f"  p_binom     = {np.max(np.abs(final_p_binom - p_binom)):.6e}",
            f"  alpha       = {np.max(np.abs(final_alphas - alphas)):.6e}",
            f"  tau         = {np.max(np.abs(final_taus - taus)):.6e}",
            "---------------------------------------"
        ])

        logger.info("\n".join(log_lines))

        return {
            "new_log_mu": final_log_mu,
            "new_alphas": final_alphas,
            "new_p_binom": final_p_binom,
            "new_taus": final_taus,
            "new_log_startprob": final_log_startprob,  # TODO
            "new_log_transmat": log_transmat,  # TODO
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -cost(res.x),
            "n_states": n_states,
        } | param_errors
    '''
    """
    def run_marg_likelihood_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat=None,
        tumor_prop=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False,  # DEPRECATE
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=1000,
        max_rdr=5.0,
        tol=1e-4,
        use_logit=True,
        propagate_errors=False,
        **kwargs,
    ):
        _, n_comp, n_spots = X.shape

        assert n_spots == 1
        assert n_comp == 2

        base_nb_mean = base_nb_mean.copy()

        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0
                base_nb_mean[est_rdr > max_rdr] = 0.0

        optimize_nb = np.any(base_nb_mean > 0)

        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

        logger.info(
            f"Solved for med. {np.median(nbEncoder.total_count):.4f} and max. {np.max(nbEncoder.total_count):.4f} total counts for nbEncoder ({nbEncoder.compression_rate:.2%} compression)."
        )
        logger.info(
            f"Solved for med. {np.median(bbEncoder.total_count):.4f} and max. {np.max(bbEncoder.total_count):.4f} total counts for bbEncoder ({bbEncoder.compression_rate:.2%} compression)."
        )

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.get_initial_params(
            n_states,
            n_spots,
            init_log_mu,
            init_p_binom,
            init_alphas,
            init_taus,
        )

        kwargs_str = (
            "{\n" + "\n".join(f"  '{k}': {v}" for k, v in kwargs.items()) + "\n}"
            if kwargs
            else "{}"
        )
        logger.info(f"Assuming kwargs={kwargs_str}")
        logger.info(
            f"Assumed initial p_binom and dispersion:\n{np.hstack((p_binom, taus))}"
        )

        x0 = self.pack_params(
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        def nll_forward(params):
            this_log_startprob, this_log_mu, this_p_binom, this_alphas, this_taus = (
                self.unpack_params(
                    params,
                    n_states,
                    log_startprob,
                    log_mu,
                    p_binom,
                    alphas,
                    taus,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
            )

            log_emission_rdr, log_emission_baf = (
                self.compute_emission_probability_nb_betabinom_coded(
                    nbEncoder,
                    bbEncoder,
                    this_log_mu,
                    this_alphas,
                    this_p_binom,
                    this_taus,
                )
            )

            log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]
            log_alpha = self.forward_lattice(
                lengths,
                log_transmat,
                this_log_startprob,
                log_emissions,
                log_sitewise_transmat,
            )

            curr = 0
            total_nll = 0
            for le in lengths:
                total_nll += -numba_logsumexp(log_alpha[:, curr + le - 1])
                curr += le

            return total_nll

        prev_x = [np.copy(x0)]

        # NB L-BFGS-B does not natively support parameter tolerance.
        def param_tol_callback(xk):
            max_change = np.max(np.abs(xk - prev_x[0]))
            prev_x[0] = np.copy(xk)

            # TODO
            if max_change < 1.0e-2:
                logger.info(
                    f"Stopping early: max parameter change ({max_change:.6e}) dropped below tolerance ({1.e-2:.6e})."
                )
                return True

        start_time_opt = time.time()
        logger.info(
            f"Starting marginal likelihood optimization with bfgs\ninitial NLL={nll_forward(x0):.6e}"
        )

        options = {
            "maxiter": kwargs.get("max_iter", max_iter),
            "ftol": 1e-6,
            "gtol": 1e-5,
            "disp": False,
        }

        logger.info(f"Assuming options={options}")

        # deepmind/optax, google/jaxopt.
        res = scipy.optimize.minimize(
            nll_forward,
            x0,
            method="L-BFGS-B",  # BFGS
            options=options,
            callback=None,  # param_tol_callback,
            bounds=self.get_bounds(
                n_states=n_states,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            ),
        )

        runtime = time.time() - start_time_opt

        logger.info(
            f"Optimization complete: {runtime:.2f}s\n"
            f"converged: {res.success}\n"
            f"message: {res.message}\n"
            f"final NLL: {res.fun:.6e}\n"
        )

        if propagate_errors:
            (log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err) = (
                self.unpack_param_errors(
                    x=res.x,
                    hess_inv=res.hess_inv,
                    n_states=n_states,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
            )

            param_errors = {
                "new_log_mu_err": log_mu_err,
                "new_alphas_err": alphas_err,
                "new_p_binom_err": p_binom_err,
                "new_taus_err": taus_err,
                "new_log_startprob_err": log_startprob_err,
            }
        else:
            param_errors = {}

        final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus = (
            self.unpack_params(
                res.x,
                n_states,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
        )

        final_rdr, final_baf = self.compute_emission_probability_nb_betabinom(
            X,
            base_nb_mean,
            final_log_mu,
            final_alphas,
            total_bb_RD,
            final_p_binom,
            final_taus,
        )

        log_emission = final_rdr + final_baf

        log_gamma = self.get_state_posteriors(
            lengths,
            log_transmat,
            final_log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))
        logger.info(
            f"Final State posterior breakdown:\n{[f'{xx:.4e}' for xx in state_prior]}"
        )

        return {
            "new_log_mu": final_log_mu,
            "new_alphas": final_alphas,
            "new_p_binom": final_p_binom,
            "new_taus": final_taus,
            "new_log_startprob": final_log_startprob,
            "new_log_transmat": log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -res.fun,
            "n_states": n_states,
        } | param_errors
    """
