from numba import njit
import numpy as np
from math import lgamma, exp
# from scipy.special import loggamma
from cnaster.count_encoder import CountEncoder


# TODO
from cnaster.hmm_nophasing import (hmm_nophasing, np_sum_ax_squeeze,
                                   numba_logsumexp, _nb_logpmf_1d, _bb_logpmf_1d)

PEANLIZE_PHASE_ONLY_ON_SAME_CNV = False  # TODO config derived.

'''
def switch_betabinom(log_emission_baf_nophase, k, n, alpha, beta):
    """
    Efficient evaluation of the _phase_ betabinomial emission given
    the unphased variety and model parameters.
    """
    # NB expect:
    #    log_emission_baf_nophase.shape=(n_states, n_obs),
    #    k.shape=(n_obs,),
    #    total_bb_RD.shape=(n_obs,),
    #    alpha.shape=(n_states,).
    return (
        log_emission_baf_nophase
        + loggamma(beta[:, np.newaxis] + k)
        - loggamma(alpha[:, np.newaxis] + k)
        + loggamma(alpha[:, np.newaxis] + n - k)
        - loggamma(beta[:, np.newaxis] + n - k)
    )
'''

@njit(nogil=True, cache=True, error_model="numpy")
def _switch_betabinom_1d(log_baf_nophase, obs, total, p_binom_s, tau_s, EPS=1e-10):
    """
    Applies the phase-switch mathematical derivation over compressed 1D arrays.
    p_binom_s and tau_s are 1D arrays of length n_states for a specific spot.
    """
    n_states, n_unique = log_baf_nophase.shape
    out = np.empty_like(log_baf_nophase)
    
    for i in range(n_states):
        alpha = max(p_binom_s[i] * tau_s[i], EPS)
        beta = max((1.0 - p_binom_s[i]) * tau_s[i], EPS)
        
        for j in range(n_unique):
            k = obs[j]
            n = total[j]
            out[i, j] = (
                log_baf_nophase[i, j]
                + lgamma(beta + k)
                - lgamma(alpha + k)
                + lgamma(alpha + n - k)
                - lgamma(beta + n - k)
            )
    return out

@njit(nogil=True, cache=True, error_model="numpy")
def update_combined_transmat(
    out_transmat,
    n_states,
    log_transmat,
    self_trans,
    switch_trans,
    penalize_phase_only_on_same_cnv,
    log_half,
):
    if penalize_phase_only_on_same_cnv:
        # NB base case: All transitions split equally between phases (0 penalty)
        out_transmat[:n_states, :n_states] = log_half + log_transmat
        out_transmat[:n_states, n_states:] = log_half + log_transmat
        out_transmat[n_states:, :n_states] = log_half + log_transmat
        out_transmat[n_states:, n_states:] = log_half + log_transmat

        # NB overwrite __only__ the diagonals of the cnv blocks (where cnv state is conserved)
        for i in range(n_states):
            # Phase 0 -> Phase 0
            out_transmat[i, i] = self_trans + log_transmat[i, i]
            # Phase 0 -> Phase 1
            out_transmat[i, i + n_states] = switch_trans + log_transmat[i, i]
            # Phase 1 -> Phase 0
            out_transmat[i + n_states, i] = switch_trans + log_transmat[i, i]
            # Phase 1 -> Phase 1
            out_transmat[i + n_states, i + n_states] = self_trans + log_transmat[i, i]
    else:
        # NB original behavior:  apply sitewise phase matrices globally
        out_transmat[:n_states, :n_states] = self_trans + log_transmat
        out_transmat[:n_states, n_states:] = switch_trans + log_transmat
        out_transmat[n_states:, :n_states] = switch_trans + log_transmat
        out_transmat[n_states:, n_states:] = self_trans + log_transmat

class hmm_phased(hmm_nophasing):
    def __init__(self, params="stmp", t=1 - 1e-4):
        super().__init__(params=params, t=t)

    @classmethod
    def compute_emission_probability_nb_betabinom(
        cls, X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, clone_stack=False
    ):
        """Builds encoders dynamically and routes to the optimized coded method."""
        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)
        
        return cls.compute_emission_probability_nb_betabinom_coded(
            nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus, clone_stack=clone_stack
        )

    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus, 
        clone_stack=True, scratch_rdr=None, scratch_baf=None
    ):
        n_states, n_spots = log_mu.shape

        n_spots = nbEncoder.n_spots
        assert n_spots == bbEncoder.n_spots, "Encoders must have identical spot counts"

        log_emit_rdr_list = []
        log_emit_baf_list = []

        for s in range(n_spots):
            nb_endog = nbEncoder.get_unique_obs(s)
            nb_exposure = nbEncoder.get_unique_total(s)
            bb_endog = bbEncoder.get_unique_obs(s)
            bb_exposure = bbEncoder.get_unique_total(s)

            log_emit_rdr_uniq = scratch_rdr[s] if scratch_rdr is not None else np.zeros((n_states, len(nb_endog)))
            log_emit_baf_uniq = scratch_baf[s] if scratch_baf is not None else np.zeros((n_states, len(bb_endog)))

            for i in range(n_states):
                _nb_logpmf_1d(
                    nb_endog, nb_exposure, exp(log_mu[i, s]), alphas[i, s], log_emit_rdr_uniq[i, :]
                )
                _bb_logpmf_1d(
                    bb_endog, bb_exposure, p_binom[i, s], taus[i, s], log_emit_baf_uniq[i, :]
                )

            # ---> NEW: Phase split happens in compressed space <---
            log_emit_baf_uniq_switched = _switch_betabinom_1d(
                log_emit_baf_uniq, bb_endog, bb_exposure, p_binom[:, s], taus[:, s]
            )

            # Stack phases (Phase 0 top, Phase 1 bottom)
            phased_rdr_uniq = np.vstack((log_emit_rdr_uniq, log_emit_rdr_uniq))
            phased_baf_uniq = np.vstack((log_emit_baf_uniq, log_emit_baf_uniq_switched))

            # Decode expanded arrays
            log_emit_rdr_list.append(nbEncoder.decode_array(phased_rdr_uniq, s))
            log_emit_baf_list.append(bbEncoder.decode_array(phased_baf_uniq, s))

        if clone_stack:
            log_emit_rdr = np.concatenate(log_emit_rdr_list, axis=1) # [:, :, np.newaxis]
            log_emit_baf = np.concatenate(log_emit_baf_list, axis=1) # [:, :, np.newaxis]
        else:
            log_emit_rdr = np.stack(log_emit_rdr_list, axis=2)
            log_emit_baf = np.stack(log_emit_baf_list, axis=2)

        return log_emit_rdr, log_emit_baf

    @staticmethod
    @njit(nogil=True, cache=True, error_model="numpy")
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
        penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV, 
    ):
        n_paired_states, n_obs, _ = log_emission.shape
        n_states = int(np.ceil(n_paired_states / 2))

        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
        log_alpha = np.zeros((n_paired_states, n_obs))
        buf = np.zeros(n_paired_states)

        log_half = np.log(0.5)
        combined_log_startprob = log_half + np.append(log_startprob, log_startprob)
        combined_transmat = np.empty((n_paired_states, n_paired_states))

        cumlen = 0
        for le in lengths:
            # Fixed np.sum across spot axis
            for j in range(n_paired_states):
                log_alpha[j, cumlen] = combined_log_startprob[j] + np.sum(log_emission[j, cumlen, :])

            for t in range(1, le):
                idx = cumlen + t - 1

                update_combined_transmat(
                    combined_transmat,
                    n_states,
                    log_transmat,
                    log_sitewise_self_transmat[idx],
                    log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv,
                    log_half,
                )

                for j in range(n_paired_states):
                    for i in range(n_paired_states):
                        buf[i] = log_alpha[i, idx] + combined_transmat[i, j]

                    log_alpha[j, cumlen + t] = numba_logsumexp(buf) + np.sum(
                        log_emission[j, cumlen + t, :]
                    )

            cumlen += le

        return log_alpha

    @staticmethod
    @njit(nogil=True, cache=True, error_model="numpy")
    def backward_lattice(
        lengths,
        log_transmat,
        log_startprob,  # Unused, but kept for API match
        log_emission,
        log_sitewise_transmat,
        penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV, 
    ):
        n_paired_states, n_obs, _ = log_emission.shape
        n_states = int(np.ceil(n_paired_states / 2))

        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
        log_beta = np.zeros((n_paired_states, n_obs))
        buf = np.zeros(n_paired_states)

        log_half = np.log(0.5)
        combined_transmat = np.empty((n_paired_states, n_paired_states))

        cumlen = 0
        for le in lengths:
            log_beta[:, cumlen + le - 1] = 0.0

            for t in range(le - 2, -1, -1):
                idx = cumlen + t

                update_combined_transmat(
                    combined_transmat,
                    n_states,
                    log_transmat,
                    log_sitewise_self_transmat[idx],
                    log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv,
                    log_half,
                )

                for i in range(n_paired_states):
                    for j in range(n_paired_states):
                        buf[j] = (
                            log_beta[j, cumlen + t + 1]
                            + combined_transmat[i, j]
                            + np.sum(log_emission[j, cumlen + t + 1, :])
                        )

                    log_beta[i, cumlen + t] = numba_logsumexp(buf)

            cumlen += le

        return log_beta

'''
class hmm_phased(hmm_nophasing):
    # TODO args, kwargs?
    def __init__(self, params="stmp", t=1 - 1e-4):
        super().__init__(params=params, t=t)

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        n_obs, _, n_spots = X.shape
        n_states = p_binom.shape[0]

        assert n_spots == 1, "TODO: remove before phased flight."

        # NB guard against
        assert (
            p_binom.shape[1] == 1
        ), f"Found p_binom={p_binom}, but expect singleton for axis=1"
        assert taus.shape[1] == 1, f"Found taus={taus}, but expect singleton for axis=1"

        # TODO delegate to parent.compute_emission_probability_nb_betabinom 
        log_emission_rdr_nophase, log_emission_baf_nophase = compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        # NB duplicate for with/out phase switch.
        log_emission_rdr = np.zeros((2 * n_states, n_obs, n_spots))
        log_emission_baf = np.zeros((2 * n_states, n_obs, n_spots))

        log_emission_rdr[:n_states, :, :] = log_emission_rdr_nophase[...]
        log_emission_rdr[n_states:, :, :] = log_emission_rdr[:n_states, :, :]

        log_emission_baf[:n_states, :, :] = log_emission_baf_nophase[...]
        log_emission_baf[n_states:, :, :] = switch_betabinom(
            log_emission_baf_nophase[...],
            X[:, 1, :],
            total_bb_RD,
            p_binom[:, 0] * taus[:, 0],
            (1.0 - p_binom[:, 0]) * taus[:, 0],
        )

        return log_emission_rdr, log_emission_baf

    @staticmethod
    def compute_emission_probability_nb_betabinom_phased_coded(
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
        n_states = p_binom.shape[0]

        # NB guard against
        assert (
            p_binom.shape[1] == 1
        ), f"Found p_binom={p_binom}, but expect singleton for axis=1"
        assert taus.shape[1] == 1, f"Found taus={taus}, but expect singleton for axis=1"

        log_emission_rdr_nophase, log_emission_baf_nophase = (
            hmm_nophasing.compute_emission_probability_nb_betabinom_coded(
                nbEncoder,
                bbEncoder,
                log_mu,
                alphas,
                p_binom,
                taus,
                clone_stack=clone_stack,
                scratch_rdr=scratch_rdr,
                scratch_baf=scratch_baf,
            )
        )

        n_obs = log_emission_rdr_nophase.shape[1]

        # NB duplicate for w/o phase switch.
        log_emission_rdr = np.zeros((2 * n_states, n_obs))
        log_emission_baf = np.zeros((2 * n_states, n_obs))

        log_emission_rdr[:n_states, :] = log_emission_rdr_nophase[:, :]
        log_emission_rdr[n_states:, :] = log_emission_rdr_nophase[:, :]

        log_emission_baf[:n_states, :] = log_emission_baf_nophase[:, :]
        log_emission_baf[n_states:, :] = switch_betabinom(
            log_emission_baf_nophase[:, :],
            bbEncoder.obs_count[:, 0],
            bbEncoder.total_count[:, 0],
            p_binom[:, 0] * taus[:, 0],
            (1.0 - p_binom[:, 0]) * taus[:, 0],
        )

        return log_emission_rdr, log_emission_baf

    @staticmethod
    @njit
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
        penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV,  # TODO config derived.
    ):
        n_paired_states = log_emission.shape[0]
        n_states = int(np.ceil(n_paired_states / 2))
        n_obs = log_emission.shape[1]

        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))

        log_alpha = np.zeros((n_paired_states, n_obs))
        buf = np.zeros(n_paired_states)

        log_half = np.log(0.5)
        combined_log_startprob = log_half + np.append(log_startprob, log_startprob)

        combined_transmat = np.empty((n_paired_states, n_paired_states))

        cumlen = 0
        for le in lengths:
            log_alpha[:, cumlen] = combined_log_startprob + np_sum_ax_squeeze(
                log_emission[:, cumlen, :], axis=1
            )

            for t in range(1, le):
                idx = cumlen + t - 1

                update_combined_transmat(
                    out_transmat=combined_transmat,
                    n_states=n_states,
                    log_transmat=log_transmat,
                    self_trans=log_sitewise_self_transmat[idx],
                    switch_trans=log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv=penalize_phase_only_on_same_cnv,
                    log_half=log_half,
                )

                for j in range(n_paired_states):
                    for i in range(n_paired_states):
                        buf[i] = log_alpha[i, idx] + combined_transmat[i, j]

                    log_alpha[j, cumlen + t] = numba_logsumexp(buf) + np.sum(
                        log_emission[j, cumlen + t, :]
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
        penalize_phase_only_on_same_cnv: bool = PEANLIZE_PHASE_ONLY_ON_SAME_CNV,  # TODO config derived.
    ):
        n_paired_states = log_emission.shape[0]
        n_states = int(np.ceil(n_paired_states / 2))
        n_obs = log_emission.shape[1]

        log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))

        log_beta = np.zeros((n_paired_states, n_obs))
        buf = np.zeros(n_paired_states)

        log_half = np.log(0.5)

        combined_transmat = np.empty((n_paired_states, n_paired_states))

        cumlen = 0
        for le in lengths:
            log_beta[:, cumlen + le - 1] = 0.0

            for t in range(le - 2, -1, -1):
                idx = cumlen + t

                update_combined_transmat(
                    out_transmat=combined_transmat,
                    n_states=n_states,
                    log_transmat=log_transmat,
                    self_trans=log_sitewise_self_transmat[idx],
                    switch_trans=log_sitewise_transmat[idx],
                    penalize_phase_only_on_same_cnv=penalize_phase_only_on_same_cnv,
                    log_half=log_half,
                )

                for i in range(n_paired_states):
                    for j in range(n_paired_states):
                        buf[j] = (
                            log_beta[j, cumlen + t + 1]
                            + combined_transmat[i, j]
                            + np.sum(log_emission[j, cumlen + t + 1, :])
                        )

                    log_beta[i, cumlen + t] = numba_logsumexp(buf)

            cumlen += le

        return log_beta
'''