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
    # convert_params_disp,
    mylogsumexp,
    np_sum_ax_squeeze,
    # get_em_solver_params,
)
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_emission_eval import compute_emissions
from cnaster.hmm_emission import nloglikeobs_nb, nloglikeobs_bb
from cnaster.hmm_sitewise import (
    compute_emission_probability_nb_betabinom_phased,
    forward_marginalize_phased,
    backward_marginalize_phased,
)
from scipy.optimize import OptimizeResult
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
        return compute_emissions(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        """
        Computes the emission probability for a compression of the
        unique instances; subsequently, broadcasting to the entire
        array.
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

    def get_initial_params(
        self,
        n_states,
        n_spots,
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
    ):
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
        use_logit=True,  # Add use_logit flag
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
            
            log_startprob_err = np.sqrt(np.clip(np.diag(cov_transformed), a_min=0, a_max=None))
            idx += n_states
        else:
            log_startprob_err = None

        if optimize_nb and "m" in self.params:
            log_mu_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            log_mu_err = None

        if "p" in self.params:
            raw_p_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
            if use_logit:
                p_binom_val = scipy.special.expit(x[idx : idx + n_states].reshape(n_states, 1))
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
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
                
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
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
                
                taus_val = np.exp(val_raw)
                taus_err = taus_val * val_err
                idx += n_states
        else:
            taus_err = None

        return log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err

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
        max_rdr=5.0,  # TODO HACK MAGIC
        tol=1e-4,
        use_logit=False,
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

        # NB solved for med. 26.0 and max. 75_849.0 total counts for bbEncoder.
        logger.info(
            f"Solved for med. {np.median(nbEncoder.total_count):.4f} and max. {np.max(nbEncoder.total_count):.4f} total counts for nbEncoder."
        )
        logger.info(
            f"Solved for med. {np.median(bbEncoder.total_count):.4f} and max. {np.max(bbEncoder.total_count):.4f} total counts for bbEncoder."
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

        # TODO HACK?
        # init_alphas = 1. / np.exp(init_log_mu) if init_log_mu is not None else init_alphas
        # init_taus = np.median(bbEncoder.total_count) * np.ones_like(p_binom)

        kwargs_str = (
            "{\n" + "\n".join(f"  '{k}': {v}" for k, v in kwargs.items()) + "\n}"
            if kwargs
            else "{}"
        )
        logger.info(f"Assuming kwargs={kwargs_str}")
        logger.info(
            f"Assumed initial p_binom and dispersion:\n{np.hstack((p_binom, taus))}"
        )

        # DEPRECATE  utilize state posterior if given.
        log_gamma = kwargs.get("log_gamma", None)

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

        self.log_startprob = log_startprob
        self.log_emissions = None
        self.state_posteriors = None
        self.iterations = 0

        # NB update state posteriors on (every other) scipy.optimize.minimize callback.
        # TODO cadence of callback?
        def update_state_posteriors(intermediate_result: OptimizeResult = None):
            # TODO m-step for log_startprob and log_transmat.
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

            self.log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]

            # NB log_gamma is (n_states * n_observations), potentially concatenated by clone on obs. axis.
            #    utilized on optimization callback.

            if self.state_posteriors is None:
                update_state_posteriors()

            # NB em cost is sum_iid of obs., sum_state of gamma * log_emission, which is negative log likelihood.
            return -np.sum(self.state_posteriors * self.log_emissions[..., 0])

        '''
        def nll_forward(params):
            this_log_startprob, this_log_mu, this_p_binom, this_alphas, this_taus = (
                self.unpack_params(
                    params,
                    n_states,
                    log_startprob,
                    log_mu,  # TODO init_log_mu
                    p_binom,
                    alphas,  # TODO init_alphas
                    taus,  # TODO init_taus
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
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

            self.log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]

            log_alpha = self.forward_lattice(
                lengths,
                log_transmat,
                this_log_startprob,
                self.log_emissions,
                log_sitewise_transmat,
            )

            curr = 0
            total_nll = 0

            for le in lengths:
                total_nll += -mylogsumexp(log_alpha[:, curr + le - 1])
                curr += le

            return total_nll
        '''

        # NB vanilla max. likelihood or baum welch.
        # cost, callback = nll_forward, None
        cost, callback = baum_welch_forward, update_state_posteriors

        start_time_opt = time.time()
        logger.info(
            f"maxlike_nb_bb with BFGS\nn_states={n_states};\nX.shape={X.shape};\nfixed_dispersion={fix_NB_dispersion};\nshared dispersion={shared_NB_dispersion};\noptimize_nb={optimize_nb};\nuse_logit={use_logit};\ninitial cost={cost(x0):.6e}"
        )

        options = {
            "maxiter": kwargs.get("max_iter", 10_000),
            # "maxfun": kwargs.get("max_fun", 5_000),
            # "gtol": 1e-6,
            # "ftol": 1e-6,
            "disp": False,
        }

        # TODO bounds
        res = scipy.optimize.minimize(
            cost,
            x0,
            method="BFGS",
            bounds=None,
            callback=callback,
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
        )

        if propagate_errors:
            # parameter_errors = np.sqrt(np.diag(res.hess_inv.todense()))
            (
                log_startprob_err, 
                log_mu_err, 
                p_binom_err, 
                alphas_err, 
                taus_err
            ) = self.unpack_param_errors(
                x=res.x,
                hess_inv=res.hess_inv,
                n_states=n_states,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit
            )

            param_errors = {
                "new_log_mu_err": log_mu_err,
                "new_alphas_err": alphas_err,
                "new_p_binom_err": p_binom_err,
                "new_taus_err": taus_err,
                "new_log_startprob_err": None, # TODO
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

        to_log = [
            f"Solved for best emission parameters with {self.__class__.__name__}:"
        ]

        if optimize_nb:
            to_log.append(f"mu=\n{[f'{xx:.3f}' for xx in final_log_mu[:,0]]}")
            to_log.append(f"alphas=\n{[f'{xx:.3f}' for xx in final_alphas[:,0]]}")

        to_log.append(f"p_binom=\n{[f'{xx:.3f}' for xx in final_p_binom[:,0]]}")
        to_log.append(f"taus=\n{[f'{xx:.3e}' for xx in final_taus[:,0]]}")

        logger.info("\n".join(to_log))

        """
        # NB emission is (nstates, n_observations, n_spots), but currently only supports n_spots=1.
        log_emission_rdr, log_emission_baf = compute_emission_probability_nb_betabinom_coded(
            nbEncoder, bbEncoder, final_log_mu, final_p_binom, final_alphas, final_taus
        )
        """

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
    
    # TODO run_marginal_like_nb_bb
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
        max_iter=1000,
        max_rdr=5.0,
        tol=1e-4,
        use_logit=False,
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
            f"Solved for med. {np.median(nbEncoder.total_count):.4f} and max. {np.max(nbEncoder.total_count):.4f} total counts for nbEncoder."
        )
        logger.info(
            f"Solved for med. {np.median(bbEncoder.total_count):.4f} and max. {np.max(bbEncoder.total_count):.4f} total counts for bbEncoder."
        )

        (
            log_mu,
            p_binom,
            alphas,
            taus,
            log_startprob,
            log_transmat,
        ) = self.get_initial_params(
            n_states, n_spots, init_log_mu, init_p_binom, init_alphas, init_taus,
        )

        kwargs_str = (
            "{\n" + "\n".join(f"  '{k}': {v}" for k, v in kwargs.items()) + "\n}"
            if kwargs else "{}"
        )
        logger.info(f"Assuming kwargs={kwargs_str}")
        logger.info(f"Assumed initial p_binom and dispersion:\n{np.hstack((p_binom, taus))}")

        # Pack parameters into a flat array for scipy
        x0 = self.pack_params(
            log_startprob, log_mu, p_binom, alphas, taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        # ---------------------------------------------------------
        # Direct Marginal Log-Likelihood Objective
        # ---------------------------------------------------------
        def nll_forward(params):
            this_log_startprob, this_log_mu, this_p_binom, this_alphas, this_taus = (
                self.unpack_params(
                    params, n_states, log_startprob, log_mu, p_binom, alphas, taus,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
            )

            # Compute emissions
            log_emission_rdr, log_emission_baf = (
                self.compute_emission_probability_nb_betabinom_coded(
                    nbEncoder, bbEncoder, this_log_mu, this_alphas, this_p_binom, this_taus,
                )
            )
            
            log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]

            # Run Forward algorithm to integrate out the hidden states exactly
            log_alpha = self.forward_lattice(
                lengths, log_transmat, this_log_startprob, log_emissions, log_sitewise_transmat,
            )

            # Sum the terminal log-probabilities for each independent contig/segment
            curr = 0
            total_nll = 0
            for le in lengths:
                total_nll += -mylogsumexp(log_alpha[:, curr + le - 1])
                curr += le

            return total_nll

        start_time_opt = time.time()
        logger.info(f"Starting Direct Marginal Likelihood Optimization with BFGS\ninitial NLL={nll_forward(x0):.6e}")

        # Removed the callback. The landscape is now perfectly stationary.
        options = {
            "maxiter": kwargs.get("max_iter", max_iter),
            "disp": False,
        }

        # We highly recommend switching to 'L-BFGS-B' if you start exceeding ~15 states
        # as dense BFGS Hessian updates become computationally expensive O(N^2).
        res = scipy.optimize.minimize(
            nll_forward,
            x0,
            method="BFGS",
            options=options,
        )

        runtime = time.time() - start_time_opt

        logger.info(
            f"Optimization complete: {runtime:.2f}s\n"
            f"converged: {res.success}\n"
            f"message: {res.message}\n"
            f"final NLL: {res.fun:.6e}\n"
        )

        # ---------------------------------------------------------
        # Finalization & Error Extraction
        # ---------------------------------------------------------
        if propagate_errors:
            (
                log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err
            ) = self.unpack_param_errors(
                x=res.x, hess_inv=res.hess_inv, n_states=n_states,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit
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
                res.x, n_states, log_startprob, log_mu, p_binom, alphas, taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion, shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion, shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
        )

        # ---------------------------------------------------------
        # Calculate Final Viterbi / Posteriors
        # ---------------------------------------------------------
        final_rdr, final_baf = self.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, final_log_mu, final_alphas, total_bb_RD, final_p_binom, final_taus,
        )

        log_emission = final_rdr + final_baf

        log_gamma = self.get_state_posteriors(
            lengths, log_transmat, final_log_startprob, log_emission, log_sitewise_transmat,
        )

        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))
        logger.info(f"Final State posterior breakdown:\n{[f'{xx:.4e}' for xx in state_prior]}")

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
    
    def optimize(self, *args, **kwargs):
        return self.run_baum_welch_nb_bb(*args, **kwargs)