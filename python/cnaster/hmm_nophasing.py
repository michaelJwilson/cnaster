import pprint
import time
from math import exp, lgamma, log

import numpy as np
import scipy.optimize
import scipy.special
from numba import njit, prange
from scipy.optimize import OptimizeResult

from cnaster.config import start_time
from cnaster.count_encoder import CountEncoder
# from cnaster.hmm_utils import compute_posterior_transition_nophasing
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


@njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def nbinom_logpmf_numba(k, r, p, parameter_terms_only=True):
    if p <= 0.0 or p >= 1.0 or r <= 0.0 or k < 0:
        return 0.0

    log_coeff = lgamma(k + r) - lgamma(r)

    if parameter_terms_only:
        log_coeff -= lgamma(k + 1)

    return log_coeff + r * log(p) + k * log(1.0 - p)


@njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def betabinom_logpmf_numba(k, n, alpha, beta, parameter_terms_only=True):
    if alpha <= 0.0 or beta <= 0.0 or n < 0 or k < 0 or k > n:
        return 0.0

    log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
    log_beta_denom = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

    if parameter_terms_only:
        log_binom_coeff = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        return log_binom_coeff + log_beta_num - log_beta_denom
    else:
        return log_beta_num - log_beta_denom


@njit(nogil=True, cache=True, error_model="numpy")
def _nb_logpmf_1d(obs, exposure, mu, alpha, out):
    # out = np.zeros_like(obs, dtype=np.float64)
    r = 1.0 / max(alpha, 1.0e-10)

    for i in range(len(obs)):
        k = obs[i]
        lambda_i = exposure[i] * mu

        if lambda_i <= 0.0:
            out[i] = 0.0
            continue

        p = 1.0 / (1.0 + alpha * lambda_i)
        out[i] = nbinom_logpmf_numba(k, r, p)

    # return out


@njit(nogil=True, cache=True, error_model="numpy")
def _bb_logpmf_1d(obs, total, p_binom, tau, out, EPS=1e-10):
    # out = np.zeros_like(obs, dtype=np.float64)
    alpha = max(p_binom * tau, EPS)
    beta = max((1.0 - p_binom) * tau, EPS)

    for i in range(len(obs)):
        out[i] = betabinom_logpmf_numba(obs[i], total[i], alpha, beta)

    # return out


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _dense_nb_logpmf(X_nb, base_nb_mean, log_mu, alphas):
    n_states = log_mu.shape[0]
    n_obs, n_spots = X_nb.shape

    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    for i in prange(n_states):
        mu_val = exp(log_mu[i, 0])
        alpha_val = alphas[i, 0]

        for s in range(n_spots):
            # out[i, :, s] = _nb_logpmf_1d(
            #     X_nb[:, s], base_nb_mean[:, s], mu_val, alpha_val
            # )
            _nb_logpmf_1d(
                X_nb[:, s], base_nb_mean[:, s], mu_val, alpha_val, out[i, :, s]
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
            # out[i, :, s] = _bb_logpmf_1d(
            #     X_bb[:, s], total_bb_RD[:, s], p_val, tau_val, EPS
            # )
            _bb_logpmf_1d(
                X_bb[:, s], total_bb_RD[:, s], p_val, tau_val, out[i, :, s], EPS
            )
    return out


@njit(cache=True, inline="always", fastmath=False, error_model="numpy")
def np_sum_ax_squeeze(arr, axis=0):
    return np.sum(arr, axis=axis)


@njit
def numba_logsumexp(a):
    a_max = np.max(a)
    if np.isinf(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


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
        nbEncoder,
        bbEncoder,
        log_mu,
        alphas,
        p_binom,
        taus,
        clone_stack=True,
        scratch_rdr=None,
        scratch_baf=None,
    ):
        n_states = log_mu.shape[0]
        n_spots = nbEncoder.n_spots

        assert bbEncoder.n_spots == n_spots

        log_emit_rdr_list, log_emit_baf_list = [], []

        for s in range(n_spots):
            nb_endog = nbEncoder.get_unique_obs(s)
            nb_exposure = nbEncoder.get_unique_total(s)

            bb_endog = bbEncoder.get_unique_obs(s)
            bb_exposure = bbEncoder.get_unique_total(s)

            log_emit_rdr_uniq = (
                scratch_rdr[s] if scratch_rdr else np.zeros((n_states, len(nb_endog)))
            )
            log_emit_baf_uniq = (
                scratch_baf[s] if scratch_baf else np.zeros((n_states, len(bb_endog)))
            )

            for i in range(n_states):
                _nb_logpmf_1d(
                    nb_endog,
                    nb_exposure,
                    exp(log_mu[i, s]),
                    alphas[i, s],
                    log_emit_rdr_uniq[i, :],
                )
                _bb_logpmf_1d(
                    bb_endog,
                    bb_exposure,
                    p_binom[i, s],
                    taus[i, s],
                    log_emit_baf_uniq[i, :],
                )

            log_emit_rdr_list.append(nbEncoder.decode_array(log_emit_rdr_uniq, s))
            log_emit_baf_list.append(bbEncoder.decode_array(log_emit_baf_uniq, s))

        if clone_stack:
            # NB concatenate clones along the genomic axis (n_states, total_obs) -> (n_states, total_obs, 1)
            log_emit_rdr = np.concatenate(
                log_emit_rdr_list, axis=1
            )  # [:, :, np.newaxis]
            log_emit_baf = np.concatenate(
                log_emit_baf_list, axis=1
            )  # [:, :, np.newaxis]
        else:
            # NB (n_states, n_obs, n_spots)
            log_emit_rdr = np.stack(log_emit_rdr_list, axis=2)
            log_emit_baf = np.stack(log_emit_baf_list, axis=2)

        return log_emit_rdr, log_emit_baf

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
    '''
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
    '''
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
        optimizer="BFGS",
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

        scratch_rdr = [
            np.zeros((n_states, len(nbEncoder.get_unique_obs(s))))
            for s in range(n_spots)
        ]
        scratch_baf = [
            np.zeros((n_states, len(bbEncoder.get_unique_obs(s))))
            for s in range(n_spots)
        ]

        if mode == "em":
            self.log_emissions, self.state_posteriors = None, None
            self.log_startprob = log_startprob
            self.iterations = 0

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
                        scratch_rdr=scratch_rdr,
                        scratch_baf=scratch_baf,
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
                        scratch_rdr=scratch_rdr,
                        scratch_baf=scratch_baf,
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
            f"Starting {mode} optimization with {optimizer}. Initial cost={cost_fn(x0):.6e}"
        )

        res = scipy.optimize.minimize(
            cost_fn,
            x0,
            method=optimizer,
            bounds=None,
            callback=callback,
            options=options,
        )

        logger.info(
            f"Optimization complete: {time.time() - start_time_opt:.2f}s | "
            f"{res.nit} iter | converged={res.success} | negative ln. likelihood={res.fun:.6e}"
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

        # TODO call coded
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
