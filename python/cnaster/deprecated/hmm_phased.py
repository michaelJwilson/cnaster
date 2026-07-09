import njit
import numpy as np
from scipy.special import loggamma

# TODO
from cnaster.hmm_nophasing import (hmm_nophasing, np_sum_ax_squeeze,
                                   numba_logsumexp)

PEANLIZE_PHASE_ONLY_ON_SAME_CNV = False  # TODO config derived.


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


@njit
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


"""
@njit
def forward_marginalize_phased(
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

                log_alpha[j, cumlen + t] = mylogsumexp(buf) + np.sum(
                    log_emission[j, cumlen + t, :]
                )

        cumlen += le

    return log_alpha
"""
"""
@njit
def backward_marginalize_phased(
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

    # Pre-allocate the buffer once
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

                log_beta[i, cumlen + t] = mylogsumexp(buf)

        cumlen += le

    return log_beta
"""
'''
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
    """
    Efficient emission probability evaluation:  calculates for the
    unique observations and then decodes to the full set.
    """
    n_states = log_mu.shape[0]

    # NB assumes a single spot, index 0.
    nb_endog = nbEncoder.get_unique_obs(0)
    nb_exposure = nbEncoder.get_unique_total(0)
    nb_valid = nb_exposure > 0

    bb_endog = bbEncoder.get_unique_obs(0)
    bb_exposure = bbEncoder.get_unique_total(0)
    bb_valid = bb_exposure > 0

    nb_ones = np.ones_like(nb_endog, dtype=float).reshape(-1, 1)
    bb_ones = np.ones_like(bb_endog, dtype=float).reshape(-1, 1)

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
        # else:
        #     log_emit_rdr_uniq[:, :] = 0.0

        if np.any(bb_valid):
            log_emit_baf_uniq[i, bb_valid] = -nloglikeobs_bb(
                bb_endog[bb_valid],
                bb_ones[bb_valid],
                bb_ones[bb_valid],
                bb_exposure[bb_valid],
                np.array([p_binom[i, 0], taus[i, 0]]),
                reduce=False,
            )
        # else:
        #     log_emit_baf_uniq[:, :] = 0.0

    log_emit_rdr = nbEncoder.decode_array(log_emit_rdr_uniq, 0)
    log_emit_baf = bbEncoder.decode_array(log_emit_baf_uniq, 0)

    return log_emit_rdr, log_emit_baf
'''
"""
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
"""


class hmm_phased(hmm_nophasing):
    # TODO args, kwargs?
    def __init__(self, params="stmp", t=1 - 1e-4):
        super().__init__(params=params, t=t)

    """
    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        return compute_emission_probability_nb_betabinom_phased(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    """

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
        return compute_emission_probability_nb_betabinom_phased_coded(
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
    """

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
