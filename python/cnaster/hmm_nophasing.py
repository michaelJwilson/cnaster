import numpy as np
import scipy.special
import scipy.optimize
import time
from cnaster.hmm_update import (
    update_emission_params_bb_nophasing_uniqvalues_mix,
    update_emission_params_nb_nophasing_uniqvalues,
    update_emission_params_nb_nophasing_uniqvalues_mix,
    update_startprob_nophasing,
    update_transition_nophasing,
)
from cnaster.hmm_utils import (
    compute_posterior_obs,
    compute_posterior_transition_nophasing,
    construct_unique_matrix,
    convert_params_disp,
    mylogsumexp,
    np_sum_ax_squeeze,
    get_em_solver_params,
)
from cnaster.hmm_emission_eval import compute_emissions
from cnaster.hmm_emission import nloglikeobs_nb, nloglikeobs_bb
from numba import njit
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


class hmm_nophasing:
    def __init__(self, params="stmp", t=1 - 1e-4):
        self.params = params
        self.t = t

    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        logger.debug(
            f"Evaluating HMRF NB+BB emission likelihood for X.shape={X.shape} and log_mu.shape={log_mu.shape}."
        )

        # LEGACY
        # return compute_emission_probability_nb_betabinom(
        #       X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        # )
        return compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

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

                if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
                    this_weighted_tp = []

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
            log_transmat: n_states * n_states. Transition probability after log transformation.
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

                    log_alpha[j, (cumlen + t)] = mylogsumexp(buf) + np.sum(
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
                    log_beta[i, (cumlen + t)] = mylogsumexp(buf)
            cumlen += le
        return log_beta

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
        return compute_posterior_obs(log_alpha, log_beta)

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

    def initialize_params(
        self,
        n_states,
        n_spots,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
    ):
        """
        Helper to initialize HMM emission and transition parameters.
        """
        # NB initialize NB logmean shift and BetaBinom prob
        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for r in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )

        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for r in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )
        # NB initialize (inverse of) dispersion param in NB and BetaBinom
        alphas = (
            0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )
        taus = 30 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        # NB initialize start probability and emission probability
        log_startprob = np.log(np.ones(n_states) / n_states)

        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1.0 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))

        return log_mu, p_binom, alphas, taus, log_startprob, log_transmat

    def pack_params(
        self,
        log_mu,
        p_binom,
        alphas,
        taus,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
    ):
        """Flatten parameters into single optimization vector."""
        params_list = []
        if optimize_nb:
            params_list.append(log_mu.flatten())
        params_list.append(p_binom.flatten())

        if optimize_nb and not fix_NB_dispersion:
            if shared_NB_dispersion:
                # Take the first element if shared (assumes initialized somewhat similarly or just takes one)
                params_list.append(np.array([np.log(alphas.flatten()[0])]))
            else:
                params_list.append(np.log(alphas.flatten()))

        if not fix_BB_dispersion:
            if shared_BB_dispersion:
                params_list.append(np.array([np.log(taus.flatten()[0])]))
            else:
                params_list.append(np.log(taus.flatten()))

        return np.concatenate(params_list)

    def unpack_params(
        self,
        x,
        n_states,
        log_mu_init,
        alphas_init,
        taus_init,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
    ):
        """Reconstruct parameter matrices from flat optimization vector."""
        idx = 0

        if optimize_nb:
            log_mu = x[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            log_mu = log_mu_init

        p_binom = np.clip(x[idx : idx + n_states].reshape(n_states, 1), 1e-6, 1 - 1e-6)
        idx += n_states

        if not optimize_nb or fix_NB_dispersion:
            alphas = alphas_init
        else:
            if shared_NB_dispersion:
                val = np.exp(x[idx])
                alphas = np.full((n_states, 1), val)
                idx += 1
            else:
                alphas = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states

        if fix_BB_dispersion:
            taus = taus_init
        else:
            if shared_BB_dispersion:
                val = np.exp(x[idx])
                taus = np.full((n_states, 1), val)
                idx += 1
            else:
                taus = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states

        return log_mu, p_binom, alphas, taus

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
        ) = self.initialize_params(
            n_states,
            n_spots,
            init_log_mu,
            init_p_binom,
            init_alphas,
            init_taus,
        )

        log_gamma = kwargs.get("log_gamma", None)

        logger.info(f"Assumed initial log_gamma?  {log_gamma is not None}")

        # NB unique_values is a list of length n_spots, each element is an array of shape (n_unique_pairs, 2) with columns of rounded (obs_count, total_count).
        #    mapping_matrices is a list of length n_spots, each element is a sparse matrix of shape (n_obs, n_unique_pairs) mapping obs. to compressed space per spot.
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
                f"----  Solving for Baum-Welch iteration {r}/{max_iter} with Neative Binomial & Beta Binomial emission  -----"
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
                # NB adjust copy-number state mu for RDR adjusted normalization.
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
                f"State posterior breakdown:\n{[xx for xx in contracted_log_gamma]}"
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
                    # NB compute mu as adjusted RDR
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

        return (
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            new_log_startprob,
            new_log_transmat,
            log_gamma,
        )

    def gibbs_minimize(
        self,
        endog,
        exposure,
        nloglikeobs_func,
        bounds,
        options,
        estep_func,
        initial_params,
        log_space=False,
        batch_frac=0.95,
    ):
        """
        Gibbs sampling-based minimization for emission parameters.
        """
        maxiter = int(options.get("maxiter", 100))

        # TODO HACK
        # current_params = np.array(initial_params)
        current_params = np.array(
            [0.199, 0.080, 0.454, 0.688, 0.450, 0.317, 0.157, 10.101]
        )

        weights = estep_func(current_params)

        # NB weights is (len(endog), K)
        num_states = weights.shape[1]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = endog / exposure
            valid_mask = (
                (exposure > 0) & (ratios > bounds[0][0]) & (ratios < bounds[0][1])
            )
            candidates = np.log(ratios[valid_mask]) if log_space else ratios[valid_mask]

        num_candidates = len(candidates)

        batch_size = (
            int(num_candidates * batch_frac)
            if batch_frac is not None
            else num_candidates
        )

        best_cost = nloglikeobs_func(endog, weights, exposure, current_params)
        best_params = current_params.copy()

        logger.info(
            f"{num_candidates} Gibbs candidates with range=({candidates.min():.4e}, {candidates.max():.4e})."
        )
        logger.info(
            f"Initial cost={best_cost:.4e} for params and posterior weights:\n{['{:.3f}'.format(xx) for xx in current_params]}\n{['{:.3f}'.format(xx) for xx in weights.mean(axis=0)]}"
        )




        for it in range(maxiter):
            batch_indices = np.random.choice(num_candidates, batch_size, replace=False)
            active_candidates = candidates[batch_indices]

            for k in range(num_states):
                # NB include current value to ensure possibility of monotonicity
                pool = np.unique(
                    np.concatenate([active_candidates, [current_params[k]]])
                )

                log_probs, pool_vals = [], []

                for cand in pool:
                    temp_params = current_params.copy()
                    temp_params[k] = cand

                    cost = nloglikeobs_func(endog, weights, exposure, temp_params)

                    log_probs.append(-cost)
                    pool_vals.append(cand)

                # NB softmax sampling
                log_probs = np.array(log_probs)

                probs = np.exp(log_probs - np.max(log_probs))
                probs /= np.sum(probs)

                old_param = current_params[k]
                # current_params[k] = np.random.choice(pool_vals, p=probs)

                def dispersion_cost(disp):
                    # NB dispersions are not in log space.
                    p = current_params.copy()
                    p[-1] = disp
                    return nloglikeobs_func(endog, weights, exposure, p)

                res = scipy.optimize.minimize_scalar(
                    dispersion_cost, bounds=bounds[-1], method="bounded"
                )

                if res.success:
                    current_params[-1] = res.x

                print(res)
                exit(0)

                weights = estep_func(current_params)

                current_cost = nloglikeobs_func(
                    endog, weights, exposure, current_params
                )

                logger.info(
                    f"Gibbs sample step for state {k}: {old_param:.4e} -> {current_params[k]:.4e} with new posterior weights=\n{['{:.3f}'.format(xx) for xx in weights.mean(axis=0)]}"
                )

                # NB sample cost, may not be monotonically decreasing, i.e. not best cost.
                logger.info(f"Gibbs iteration {it}: cost={current_cost:.4e}")

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()

                    logger.info(
                        f"Found new best cost={current_cost:.4e} with best params=\n{best_params}"
                    )

        exit(0)

        return scipy.optimize.OptimizeResult(x=best_params, fun=best_cost, success=True)

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
        _, n_comp, n_spots = X.shape
        assert n_spots == 1
        assert n_comp == 2

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.initialize_params(
            n_states,
            n_spots,
            init_log_mu,
            init_p_binom,
            init_alphas,
            init_taus,
        )

        unique_vals_nb, maps_nb = construct_unique_matrix(X[:, 0, :], base_nb_mean)
        unique_vals_bb, maps_bb = construct_unique_matrix(X[:, 1, :], total_bb_RD)

        nb_mapper, bb_mapper = maps_nb[0], maps_bb[0]
        nb_endog, nb_exp = unique_vals_nb[0][:, 0], unique_vals_nb[0][:, 1]
        bb_endog, bb_exp = unique_vals_bb[0][:, 0], unique_vals_bb[0][:, 1]

        optimize_nb = np.any(nb_exp > 0)

        # NB compute log_emission for (n_states, unique n_obs) and broadcasts to (n_states, n_obs) with mapper.
        def compute_log_emissions(c_log_mu, c_p_binom, c_alphas, c_taus):
            idx_nz = nb_exp > 0
            log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
            if optimize_nb and np.any(idx_nz):
                exog_dummy = np.ones((np.sum(idx_nz), 1))
                w_dummy = np.ones(np.sum(idx_nz))

                for i in range(n_states):
                    log_emit_rdr_uniq[i, idx_nz] = -nloglikeobs_nb(
                        nb_endog[idx_nz],
                        exog_dummy,
                        w_dummy,
                        nb_exp[idx_nz],
                        np.array([c_log_mu[i, 0], c_alphas[i, 0]]),
                        reduce=False,
                    )

            log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))
            exog_dummy = np.ones((len(bb_endog), 1))
            w_dummy = np.ones(len(bb_endog))
            for i in range(n_states):
                log_emit_baf_uniq[i, :] = -nloglikeobs_bb(
                    bb_endog,
                    exog_dummy,
                    w_dummy,
                    bb_exp,
                    np.array([c_p_binom[i, 0], c_taus[i, 0]]),
                    reduce=False,
                )

            log_emit_rdr = log_emit_rdr_uniq @ nb_mapper.T
            log_emit_baf = log_emit_baf_uniq @ bb_mapper.T
            return (log_emit_rdr + log_emit_baf)[:, :, np.newaxis]

        def params_to_estep(curr_log_mu, curr_p_binom, curr_alphas, curr_taus):
            log_emit = compute_log_emissions(
                curr_log_mu, curr_p_binom, curr_alphas, curr_taus
            )
            return self.get_state_posteriors(
                lengths, log_transmat, log_startprob, log_emit, log_sitewise_transmat
            )

        def solve_component(
            endog, exp, mapper, init_p, bounds, is_log, nll_func, update_params_func
        ):
            def estep_adapter(p_flat):
                curr_p = update_params_func(p_flat)
                log_gamma = params_to_estep(*curr_p)

                return (np.exp(log_gamma) @ mapper).T

            def nll_adapter(endog, weights, exposure, p_flat):
                state_params = p_flat[:-1]
                disp_param = p_flat[-1]  # Always natural scale now

                total_nll = 0
                exog_ones = np.ones((len(endog), 1))

                for k in range(n_states):
                    # For NB state_params are log_mu (is_log=True), for BB they are p (is_log=False)
                    # disp_param is alpha or tau (natural scale)

                    k_vec = np.array([state_params[k], disp_param])
                    total_nll += nll_func(
                        endog, exog_ones, weights[:, k], exposure, k_vec
                    )
                return total_nll

            return self.gibbs_minimize(
                endog,
                exp,
                nll_adapter,
                bounds,
                kwargs,
                estep_adapter,
                init_p,
                log_space=is_log,
            )

        if optimize_nb:
            logger.info("max_gibbs: optimizing nb...")

            def update_nb_params(p_flat):
                # p_flat -> (log_mu, p_binom, alphas, taus)
                new_mus = p_flat[:-1].reshape(n_states, 1)
                new_alphas = np.full((n_states, 1), p_flat[-1])  # p_flat[-1] is alpha
                return new_mus, p_binom, new_alphas, taus

            # Pass natural alpha
            init_nb = np.concatenate([log_mu.flatten(), [alphas.flatten()[0]]])
            res_nb = solve_component(
                nb_endog,
                nb_exp,
                nb_mapper,
                init_nb,
                (
                    [(-10, 10), (1e-4, 100)]
                    if fix_NB_dispersion
                    else [(-10, 10), (1e-4, 100)]
                ),
                True,  # is_log refers to mu only
                nloglikeobs_nb,
                update_nb_params,
            )

            log_mu, _, alphas, _ = update_nb_params(res_nb.x)

        logger.info("max_gibbs: optimizing bb ...")

        def update_bb_params(p_flat):
            new_ps = p_flat[:-1].reshape(n_states, 1)

            # NB dispersions are not in log space.
            new_taus = np.full((n_states, 1), p_flat[-1])
            return log_mu, new_ps, alphas, new_taus

        # Pass natural tau
        init_bb = np.concatenate([p_binom.flatten(), [taus.flatten()[0]]])
        res_bb = solve_component(
            bb_endog,
            bb_exp,
            bb_mapper,
            init_bb,
            [(0.01, 0.99), (1, 1_000)],
            False,
            nloglikeobs_bb,
            update_bb_params,
        )

        _, p_binom, _, taus = update_bb_params(res_bb.x)
        log_gamma = params_to_estep(log_mu, p_binom, alphas, taus)

        exit(0)

        return log_mu, alphas, p_binom, taus, log_startprob, log_transmat, log_gamma

    def run_max_like_nb_bb(
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
        Maximizes likelihood using scipy.optimize.minimize on the negative log-likelihood
        calculated via a forward pass. Parameters are flattened for the optimizer.
        """
        _, n_comp, n_spots = X.shape
        assert n_spots == 1
        assert n_comp == 2

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.initialize_params(
            n_states,
            n_spots,
            init_log_mu,
            init_p_binom,
            init_alphas,
            init_taus,
        )

        unique_values_nb, mapping_matrices_nb = construct_unique_matrix(
            X[:, 0, :], base_nb_mean
        )
        unique_values_bb, mapping_matrices_bb = construct_unique_matrix(
            X[:, 1, :], total_bb_RD
        )

        # NB defined for one-spot online, first element of list can be extracted.
        nb_mapper = mapping_matrices_nb[0]
        bb_mapper = mapping_matrices_bb[0]

        bb_endog = unique_values_bb[0][:, 0]
        bb_exposure = unique_values_bb[0][:, 1]

        nb_endog = unique_values_nb[0][:, 0]
        nb_exposure = unique_values_nb[0][:, 1]

        optimize_nb = np.any(nb_exposure > 0)

        x0 = self.pack_params(
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
        )

        def compute_log_emissions(this_log_mu, this_p_binom, this_alphas, this_taus):
            log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
            log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

            exog_nb = np.ones((len(nb_endog), 1))
            weights_nb = np.ones(len(nb_endog))
            exog_bb = np.ones((len(bb_endog), 1))
            weights_bb = np.ones(len(bb_endog))

            idx_nonzero_mean = nb_exposure > 0

            for i in range(n_states):
                if np.any(idx_nonzero_mean):
                    log_emit_rdr_uniq[i, idx_nonzero_mean] = -nloglikeobs_nb(
                        nb_endog[idx_nonzero_mean],
                        exog_nb[idx_nonzero_mean],
                        weights_nb[idx_nonzero_mean],
                        nb_exposure[idx_nonzero_mean],
                        np.array([this_log_mu[i, 0], this_alphas[i, 0]]),
                        reduce=False,
                    )

                log_emit_baf_uniq[i, :] = -nloglikeobs_bb(
                    bb_endog,
                    exog_bb,
                    weights_bb,
                    bb_exposure,
                    np.array([this_p_binom[i, 0], this_taus[i, 0]]),
                    reduce=False,
                )

            log_emit_rdr = log_emit_rdr_uniq @ nb_mapper.T
            log_emit_baf = log_emit_baf_uniq @ bb_mapper.T
            return (log_emit_rdr + log_emit_baf)[:, :, np.newaxis]

        def nll_forward(params):
            this_log_mu, this_p_binom, this_alphas, this_taus = self.unpack_params(
                params,
                n_states,
                log_mu,
                alphas,
                taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
            )

            log_emission = compute_log_emissions(
                this_log_mu, this_p_binom, this_alphas, this_taus
            )

            log_alpha = self.forward_lattice(
                lengths,
                log_transmat,
                log_startprob,
                log_emission,
                log_sitewise_transmat,
            )

            curr = 0
            total_nll = 0

            # TODO TBC
            for le in lengths:
                total_nll += -mylogsumexp(log_alpha[:, curr + le - 1])
                curr += le

            return total_nll

        start_time_opt = time.time()
        logger.info(
            f"maxlike_nb_bb with\n\tn_states={n_states};\n\tX.shape={X.shape});\n\tfixed_dispersion={fix_NB_dispersion};\n\tshared dispersion={shared_NB_dispersion};\n\toptimize_nb={optimize_nb};\n\tinitial nloglike={nll_forward(x0):.6e} @ start_params:\n{[f'{xx:.3f}' for xx in x0]}"
        )

        # TODO HACK
        # options = get_em_solver_params()
        options = {"maxiter": kwargs.get("max_iter", 1_000), "disp": False}

        res = scipy.optimize.minimize(
            nll_forward,
            x0,
            method="BFGS",
            options=options,
        )

        end_time_opt = time.time()
        runtime = end_time_opt - start_time_opt

        logger.info(
            f"maxlike_nb_bb complete: {runtime:.2f}s with BFGS\nX_shape={X.shape},\n"
            f"{len(x0)} params,\n"
            f"{res.nit} iter,\n"
            f"{res.nfev} fcalls,\n"
            f"converged: {res.success},\n"
            f"message: {res.message},\n"
            f"nll: {res.fun:.6e}\n"
            f"params:\n{[f'{float(xx):.3f}' for xx in res.x]}"
        )

        # TODO
        # NB defaults to init parameters for any not solved for.
        final_log_mu, final_p_binom, final_alphas, final_taus = self.unpack_params(
            res.x,
            n_states,
            log_mu,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
        )

        log_emission = compute_log_emissions(
            final_log_mu, final_p_binom, final_alphas, final_taus
        )

        log_gamma = self.get_state_posteriors(
            lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
        )

        try:
            # NB marginalized errors per parameter as per eqn. (20) of Heavens,
            #    see https://arxiv.org/pdf/0906.0664
            #
            hess_inv = res.hess_inv
            errs = 2.0 * np.sqrt(np.diag(hess_inv))
        except Exception as e:
            logger.warning(f"Could not compute parameter errors from Hessian: {e}")
            errs = None

        if errs is not None:
            safe_x = np.where(np.abs(res.x) < 1e-9, 1e-9, res.x)
            frac_errors_pct = (errs / np.abs(safe_x)) * 100
            formatted_params_str = "\n".join(
                [
                    f"{float(val):8.3f} +/- {float(err):8.3f} (95% confidence, {float(pct):5.1f}% frac. error)"
                    for val, err, pct in zip(res.x, errs, frac_errors_pct)
                ]
            )

            logger.info(f"Parameter estimates:\n{formatted_params_str}")

        exit(0)

        return (
            final_log_mu,
            final_alphas,
            final_p_binom,
            final_taus,
            log_startprob,
            log_transmat,
            log_gamma,
        )
