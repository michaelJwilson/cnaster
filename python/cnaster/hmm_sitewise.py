import numpy as np
import scipy.stats
from scipy.special import loggamma
from cnaster.hmm_update import (
    update_emission_params_bb_sitewise_uniqvalues,
    update_emission_params_bb_sitewise_uniqvalues_mix,
    update_emission_params_nb_sitewise_uniqvalues,
    update_emission_params_nb_sitewise_uniqvalues_mix,
    update_startprob_sitewise,
    update_transition_sitewise,
)
from cnaster.hmm_emission_eval import compute_emissions
from cnaster.hmm_utils import (
    compute_posterior_obs,
    compute_posterior_transition_sitewise,
    construct_unique_matrix,
    mylogsumexp,
    np_sum_ax_squeeze,
)
from numba import njit
# from cnaster.hmm_emission_eval import compute_emissions
# from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def switch_betabinom(log_emission_baf_nophase, k, n, alpha, beta):
    # NB expect: 
    #    log_emission_baf_nophase.shape=(n_states, n_obs, n_spots),
    #    k.shape=(n_obs, n_spots),
    #    total_bb_RD.shape=(n_obs, n_spots),
    #    alpha.shape=(n_states, 1).
    return (
        log_emission_baf_nophase
        + loggamma(beta[:, np.newaxis, np.newaxis] + k)
        - loggamma(alpha[:, np.newaxis, np.newaxis] + k)
        + loggamma(alpha[:, np.newaxis, np.newaxis] + n - k)
        - loggamma(beta[:, np.newaxis, np.newaxis] + n - k)
    )

def compute_emission_probability_nb_betabinom_phased(
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
):
    n_obs, _, n_spots = X.shape
    n_states = p_binom.shape[0]

    assert n_spots == 1

    # NB guard against 
    assert p_binom.shape[1] == 1, f"Found p_binom={p_binom}, but expect singleton for axis=1"
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

@njit
def forward_marginalize_phased(
    lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
):
    """
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_alpha: size 2n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
    """
    n_states = int(np.ceil(log_emission.shape[0] / 2))
    n_obs = log_emission.shape[1]

    assert (
        np.sum(lengths) == n_obs
    ), "Sum of lengths must be equal to the first dimension of X!"

    assert (
        log_startprob.shape[0] == n_states
    ), f"startprob.shape={log_startprob.shape} does match n_states={n_states} expectation given log_emission.shape={log_emission.shape}."

    assert log_emission.shape[0] % 2 == 0, f"Expect 2 * n_states for phasing;  detected odd {log_emission.shape}"

    log_sitewise_self_transmat = np.log(1. - np.exp(log_sitewise_transmat))

    log_alpha = np.zeros((log_emission.shape[0], n_obs))
    buf = np.zeros(log_emission.shape[0])
    cumlen = 0

    # NB split prob. equally across phases, i.e. half each.
    combined_log_startprob = np.log(0.5) + np.append(
        log_startprob, log_startprob
    )

    for le in lengths:
        log_alpha[:, cumlen] = combined_log_startprob + np_sum_ax_squeeze(
            log_emission[:, cumlen, :], axis=1
        )

        for t in np.arange(1, le):
            log_phases_switch_mat = np.array(
                [
                    [
                        log_sitewise_self_transmat[cumlen + t - 1],
                        log_sitewise_transmat[cumlen + t - 1],
                    ],
                    [
                        log_sitewise_transmat[cumlen + t - 1],
                        log_sitewise_self_transmat[cumlen + t - 1],
                    ],
                ]
            )
            # NEW 
            combined_transmat = np.empty((2 * n_states, 2 * n_states))
            combined_transmat[:n_states, :n_states] = log_phases_switch_mat[0, 0] + log_transmat
            combined_transmat[:n_states, n_states:] = log_phases_switch_mat[0, 1] + log_transmat
            combined_transmat[n_states:, :n_states] = log_phases_switch_mat[1, 0] + log_transmat
            combined_transmat[n_states:, n_states:] = log_phases_switch_mat[1, 1] + log_transmat

            for j in np.arange(log_emission.shape[0]):
                for i in np.arange(log_emission.shape[0]):
                    buf[i] = (
                        log_alpha[i, (cumlen + t - 1)] + combined_transmat[i, j]
                    )

                log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(
                    log_emission[j, (cumlen + t), :]
                )

        cumlen += le

    return log_alpha

@njit
def backward_marginalize_phased(
    lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
):
    """
    Note that n_states is the CNV states, and there are 2 * n_states of paired states for (CNV, phasing) pairs.
    Input
        X: size n_observations * n_components * n_spots.
        lengths: sum of lengths = n_observations.
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_startprob: n_states. Start probability after log transformation.
        log_emission: 2*n_states * n_observations * n_spots. Log probability.
        log_sitewise_transmat: n_observations, the log transition probability of phase switch.
    Output
        log_beta: size 2*n_states * n_observations. log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
    """
    n_obs = log_emission.shape[1]
    n_states = int(np.ceil(log_emission.shape[0] / 2))
    assert (
        np.sum(lengths) == n_obs
    ), "Sum of lengths must be equal to the first dimension of X!"
    assert (
        len(log_startprob) == n_states
    ), f"startprob.shape={log_startprob.shape} does match expectation given log_emission.shape={log_emission.shape}."

    assert log_emission.shape[0] % 2 == 0, f"Expect 2 * n_states for phasing;  detected odd {log_emission.shape}"

    log_sitewise_self_transmat = np.log(1. - np.exp(log_sitewise_transmat))

    log_beta = np.zeros((log_emission.shape[0], n_obs))
    buf = np.zeros(log_emission.shape[0])
    cumlen = 0
    for le in lengths:
        log_beta[:, (cumlen + le - 1)] = 0

        for t in np.arange(le - 2, -1, -1):
            log_phases_switch_mat = np.array(
                [
                    [
                        log_sitewise_self_transmat[cumlen + t],
                        log_sitewise_transmat[cumlen + t],
                    ],
                    [
                        log_sitewise_transmat[cumlen + t],
                        log_sitewise_self_transmat[cumlen + t],
                    ],
                ]
            )

            # NEW 
            combined_transmat = np.empty((2 * n_states, 2 * n_states))
            combined_transmat[:n_states, :n_states] = log_phases_switch_mat[0, 0] + log_transmat
            combined_transmat[:n_states, n_states:] = log_phases_switch_mat[0, 1] + log_transmat
            combined_transmat[n_states:, :n_states] = log_phases_switch_mat[1, 0] + log_transmat
            combined_transmat[n_states:, n_states:] = log_phases_switch_mat[1, 1] + log_transmat
            
            for i in np.arange(log_emission.shape[0]):
                for j in np.arange(log_emission.shape[0]):
                    buf[j] = (
                        log_beta[j, (cumlen + t + 1)]
                        + combined_transmat[i, j]
                        + np.sum(log_emission[j, (cumlen + t + 1), :])
                    )
                log_beta[i, (cumlen + t)] = mylogsumexp(buf)
        cumlen += le
    return log_beta

class hmm_sitewise:
    def __init__(self, params="stmp", t=1.0 - 1.0e-4):
        self.params = params
        self.t = t

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        return compute_emission_probability_nb_betabinom_phased(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )
    
    @staticmethod
    @njit
    def forward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        return forward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )

    @staticmethod
    def backward_lattice(
        lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        return backward_marginalize_phased(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )

    def run_baum_welch_nb_bb(
        self,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
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
    ):
        _, _, n_spots = X.shape

        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for _ in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )

        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for _ in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )

        alphas = (
            0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )

        taus = 30.0 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        log_startprob = np.log(np.ones(n_states) / n_states)

        # NB phased specific definition.
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1.0 - self.t) / (n_states - 1.0)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))

        logger.info("Constructing NB compression in (X[:, 0, :], base_nb_mean).")

        # NB latter is all zero for initial BAF only runs.
        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(
            X[:, 0, :], base_nb_mean
        )

        logger.info("Constructing BB compression in (X[:, 1, :], total_bb_RD).")

        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(
            X[:, 1, :], total_bb_RD
        )

        for r in range(max_iter):
            logger.info(
                f"----  Solving for Baum-Welch iteration {r}/{max_iter} with NegBin+BetaBin emission  -----"
            )

            if tumor_prop is None:
                (
                    log_emission_rdr,
                    log_emission_baf,
                ) = hmm_sitewise.compute_emission_probability_nb_betabinom(
                    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                )
            else:
                (
                    log_emission_rdr,
                    log_emission_baf,
                ) = hmm_sitewise.compute_emission_probability_nb_betabinom_mix(
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

            log_alpha = hmm_sitewise.forward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_beta = hmm_sitewise.backward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            log_gamma = compute_posterior_obs(log_alpha, log_beta)

            logger.info(
                f"State posterior breakdown:\n{[xx for xx in np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))]}"
            )

            # ----  M step  ----
            if "s" in self.params:
                new_log_startprob = update_startprob_sitewise(
                    lengths, log_gamma
                ).flatten()
            else:
                new_log_startprob = log_startprob

            if "t" in self.params:
                log_xi = compute_posterior_transition_sitewise(
                    log_alpha, log_beta, log_transmat, log_emission
                )

                new_log_transmat = update_transition_sitewise(log_xi, is_diag=is_diag)
            else:
                new_log_transmat = log_transmat

            # TODO? logmu_shift?
            if "m" in self.params:
                if tumor_prop is None:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_sitewise_uniqvalues(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        base_nb_mean,
                        alphas,
                        start_log_mu=log_mu,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                    )
                else:
                    (
                        new_log_mu,
                        new_alphas,
                    ) = update_emission_params_nb_sitewise_uniqvalues_mix(
                        unique_values_nb,
                        mapping_matrices_nb,
                        log_gamma,
                        base_nb_mean,
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
                    ) = update_emission_params_bb_sitewise_uniqvalues(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        total_bb_RD,
                        taus,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
                else:
                    (
                        new_p_binom,
                        new_taus,
                    ) = update_emission_params_bb_sitewise_uniqvalues_mix(
                        unique_values_bb,
                        mapping_matrices_bb,
                        log_gamma,
                        total_bb_RD,
                        taus,
                        tumor_prop,
                        start_p_binom=p_binom,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                    )
            else:
                new_p_binom = p_binom
                new_taus = taus

            logger.info(
                "Found max HMM parameter updates for tol=%.6e: \nstart prob.=%.6e\ntransfer matrix=%.6e\nlog_mu=%.6e\np_binom=%.6e",
                tol,
                np.max(np.abs(np.exp(new_log_startprob) - np.exp(log_startprob))),
                np.max(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))),
                np.max(np.abs(new_log_mu - log_mu)),
                np.max(np.abs(new_p_binom - p_binom)),
            )

            transmat_converged = (
                np.mean(np.abs(np.exp(new_log_transmat) - np.exp(log_transmat))) < tol
            )
            log_mu_converged = np.mean(np.abs(new_log_mu - log_mu)) < tol
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

        return {
            "new_log_mu": new_log_mu,
            "new_alphas": new_alphas,
            "new_p_binom": new_p_binom,
            "new_taus": new_taus,
            "new_log_startprob": log_startprob,  # TODO
            "new_log_transmat": log_transmat,  # TODO
            "log_gamma": log_gamma,
        }