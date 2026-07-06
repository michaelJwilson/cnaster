import numpy as np
import scipy
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
        n_states = log_mu.shape[0]

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

    @staticmethod
    @njit
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
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
        alphas = (
            0.1 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )
        taus = 30 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        log_startprob = np.log(np.ones(n_states) / n_states)

        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1.0 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))

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
        max_tau=10_000.0, # The cap where the Beta-Binomial becomes practically Binomial
    ):
        """
        Dynamically constructs the bounds list of (min, max) tuples to perfectly 
        match the flattened optimization vector generated by pack_params.
        """
        bounds = []

        # 1. Start Probabilities
        if "s" in self.params:
            # Unconstrained because softmax is applied later
            bounds.extend([(None, None)] * n_states)

        # 2. Log_mu
        if optimize_nb and "m" in self.params:
            bounds.extend([(-10.0, 10.0)] * n_states)

        # 3. Probabilities (p_binom)
        if "p" in self.params:
            if use_logit:
                # Logit space (-inf, inf) naturally bounds the probability to (0, 1)
                bounds.extend([(None, None)] * n_states)
            else:
                # Hard bounds if optimizing in linear space
                bounds.extend([(1.0e-6, 1.0 - 1.0e-6)] * n_states)

        # 4. NB Dispersion (log(alpha))
        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            # We optimize x = log(alpha). Bounding between log(1e-6) and log(100k) 
            alpha_bounds = (-13.8, 11.5)
            if shared_NB_dispersion:
                bounds.append(alpha_bounds)
            else:
                bounds.extend([alpha_bounds] * n_states)

        # 5. BB Dispersion (log(rho) where rho = 1/tau)
        if "p" in self.params and not fix_BB_dispersion:
            # We optimize x = log(1/tau). 
            # To cap tau at max_tau, we set the LOWER bound of x to log(1 / max_tau).
            min_log_rho = float(np.log(1.0 / max_tau))
            # Upper bound set to 11.5 (tau ~ 1e-5, extremely overdispersed)
            tau_bounds = (min_log_rho, 11.5)
            
            if shared_BB_dispersion:
                bounds.append(tau_bounds)
            else:
                bounds.extend([tau_bounds] * n_states)

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
            # MAP TAU -> RHO (1/tau) internally to optimize near 0 rather than inf
            if shared_BB_dispersion:
                params_list.append(np.array([np.log(1.0 / taus.flatten()[0])]))
            else:
                params_list.append(np.log(1.0 / taus.flatten()))

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
            # MAP RHO -> TAU externally
            if shared_BB_dispersion:
                val = 1.0 / np.exp(x[idx])
                taus = np.full((n_states, 1), val)
                idx += 1
            else:
                taus = (1.0 / np.exp(x[idx : idx + n_states])).reshape(n_states, 1)
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
            raw_p_val = x[idx : idx + n_states].reshape(n_states, 1)
            if use_logit:
                p_binom_val = scipy.special.expit(raw_p_val)
                p_binom_err = p_binom_val * (1 - p_binom_val) * raw_p_err
                
                # Mask bounded values where gradient naturally vanishes
                # p_binom_err[(p_binom_val <= 1e-5) | (p_binom_val >= 1 - 1e-5)] = np.nan
            else:
                p_binom_err = raw_p_err
                
                # Mask values stuck at hard clip boundaries
                p_binom_err[(raw_p_val <= 1e-5) | (raw_p_val >= 1 - 1e-5)] = np.nan
            idx += n_states
        else:
            p_binom_err = None

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]
                
                alpha_val = np.exp(val_raw)
                alpha_err = alpha_val * val_err if alpha_val > 1e-5 else np.nan
                alphas_err = np.full((n_states, 1), alpha_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
                
                alphas_val = np.exp(val_raw)
                alphas_err = alphas_val * val_err
                alphas_err[alphas_val < 1e-5] = np.nan
                idx += n_states
        else:
            alphas_err = None

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]
                
                tau_val = 1.0 / np.exp(val_raw)
                tau_err = tau_val * val_err if tau_val < 5000 else np.nan
                taus_err = np.full((n_states, 1), tau_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
                
                taus_val = 1.0 / np.exp(val_raw)
                taus_err = taus_val * val_err
                taus_err[taus_val > 5000] = np.nan
                idx += n_states
        else:
            taus_err = None

        return log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err

    # (Skipped the original un-utilized EM implementation block for brevity, 
    # it remains structurally identical)

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
        max_rdr=5.0,  
        tol=1e-4,
        use_logit=True,  # DEFAULTS TO TRUE FOR BETTER HESSIAN CONDITIONING
        propagate_errors=False,
        **kwargs,
    ):
        _, n_comp, n_spots = X.shape

        assert n_spots == 1
        assert n_comp == 2

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

        log_gamma = kwargs.get("log_gamma", None)

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

        def update_state_posteriors(intermediate_result: OptimizeResult = None):
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
            _, this_log_mu, this_p_binom, this_alphas, this_taus = self.unpack_params(
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

            self.log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]

            if self.state_posteriors is None:
                update_state_posteriors()

            return -np.sum(self.state_posteriors * self.log_emissions[..., 0])

        cost, callback = baum_welch_forward, update_state_posteriors

        start_time_opt = time.time()
        logger.info(
            f"maxlike_nb_bb with BFGS\nn_states={n_states};\nX.shape={X.shape};\nfixed_dispersion={fix_NB_dispersion};\nshared dispersion={shared_NB_dispersion};\noptimize_nb={optimize_nb};\nuse_logit={use_logit};\ninitial cost={cost(x0):.6e}"
        )

        options = {
            "maxiter": kwargs.get("max_iter", 10_000),
            "disp": False,
        }

        bounds = self.get_bounds(
            n_states=n_states,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
            max_tau=10_000.0, # TODO
        )

        # method, bounds = "BFGS", None
        method, bounds = "L-BFGS-B", bounds

        res = scipy.optimize.minimize(
            cost,
            x0,
            method=method,
            bounds=bounds,
            callback=callback,
            options=options,
        )

        end_time_opt = time.time()
        runtime = end_time_opt - start_time_opt

        logger.info(
            f"maxlike_nb_bb complete: {runtime:.2f}s with {method}\nX_shape={X.shape},\n"
            f"{len(x0)} params,\n"
            f"{res.nit} iter,\n"
            f"{res.nfev} fcalls,\n"
            f"converged: {res.success},\n"
            f"message: {res.message},\n"
            f"nll: {res.fun:.6e}\n"
        )

        if propagate_errors:
            """
            hessian_estimate = scipy.optimize.approx_derivative(
                cost,          
                res.x,
                method='3-point' 
            )
            
            # NB pseudo-inverse for stability, if the matrix is ill-conditioned
            inv_hessian_estimate = np.linalg.pinv(hessian_estimate)
            """
            (
                log_startprob_err, 
                log_mu_err, 
                p_binom_err, 
                alphas_err, 
                taus_err
            ) = self.unpack_param_errors(
                x=res.x,
                hess_inv=res.hess_inv.todense(), # L-BFGS-B requires sparse cast; previously res.hess_inv
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

        to_log = [
            f"Solved for best emission parameters with {self.__class__.__name__}:"
        ]

        if optimize_nb:
            to_log.append(f"mu=\n{[f'{xx:.3f}' for xx in final_log_mu[:,0]]}")
            to_log.append(f"alphas=\n{[f'{xx:.3f}' for xx in final_alphas[:,0]]}")

        to_log.append(f"p_binom=\n{[f'{xx:.3f}' for xx in final_p_binom[:,0]]}")
        to_log.append(f"taus=\n{[f'{xx:.3e}' for xx in final_taus[:,0]]}")

        logger.info("\n".join(to_log))

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
            "new_log_startprob": final_log_startprob,  
            "new_log_transmat": log_transmat,  
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -cost(res.x),
            "n_states": n_states,
        } | param_errors