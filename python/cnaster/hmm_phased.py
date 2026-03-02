import numpy as np
from scipy.special import loggamma
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_emission import nloglikeobs_nb, nloglikeobs_bb
from cnaster.hmm_sitewise import (
    compute_emission_probability_nb_betabinom_phased,
    forward_marginalize_phased,
    backward_marginalize_phased,
)

def switch_betabinom(log_emission_baf_nophase, k, n, alpha, beta):
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

def compute_emission_probability_nb_betabinom_coded(
    nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
):
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

    log_emit_rdr = nbEncoder.decode_array(log_emit_rdr_uniq, 0)
    log_emit_baf = bbEncoder.decode_array(log_emit_baf_uniq, 0)

    return log_emit_rdr, log_emit_baf


def compute_emission_probability_nb_betabinom_phased_coded(
    nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
):
    n_states = p_binom.shape[0]

    # NB guard against
    assert (
        p_binom.shape[1] == 1
    ), f"Found p_binom={p_binom}, but expect singleton for axis=1"
    assert taus.shape[1] == 1, f"Found taus={taus}, but expect singleton for axis=1"

    log_emission_rdr_nophase, log_emission_baf_nophase = (
        compute_emission_probability_nb_betabinom_coded(
            nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
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


class hmm_phased(hmm_nophasing):
    def __init__(self, params="stmp", t=1 - 1e-4):
        super().__init__(params=params, t=t)

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        return compute_emission_probability_nb_betabinom_phased(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        return compute_emission_probability_nb_betabinom_phased_coded(
            nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
        )

    @staticmethod
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        return forward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )

    @staticmethod
    def backward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        return backward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )
