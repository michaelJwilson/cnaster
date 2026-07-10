import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import pprint
import time
import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeResult

from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


class hmm_nophasing_jax(hmm_nophasing):
    @staticmethod
    def jax_unpack(
        params,
        params_str,
        n_states,
        optimize_nb,
        fix_NB_dispersion,
        shared_NB_dispersion,
        fix_BB_dispersion,
        shared_BB_dispersion,
        use_logit,
        init_log_startprob,
        init_log_mu,
        init_p_binom,
        init_alphas,
        init_taus,
    ):
        idx = 0
        if "s" in params_str:
            raw_startprob = params[idx : idx + n_states]
            log_startprob = raw_startprob - jsp.logsumexp(raw_startprob)
            idx += n_states
        else:
            log_startprob = jnp.array(init_log_startprob)

        if optimize_nb and "m" in params_str:
            log_mu = params[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            log_mu = jnp.array(init_log_mu)

        if "p" in params_str:
            if use_logit:
                p_binom = jsp.expit(params[idx : idx + n_states].reshape(n_states, 1))
            else:
                p_binom = jnp.clip(
                    params[idx : idx + n_states].reshape(n_states, 1), 1e-6, 1 - 1e-6
                )
            idx += n_states
        else:
            p_binom = jnp.array(init_p_binom)

        if optimize_nb and "m" in params_str and not fix_NB_dispersion:
            if shared_NB_dispersion:
                alphas = jnp.full((n_states, 1), jnp.exp(params[idx]))
                idx += 1
            else:
                alphas = jnp.exp(params[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            alphas = jnp.array(init_alphas)

        if "p" in params_str and not fix_BB_dispersion:
            if shared_BB_dispersion:
                taus = jnp.full((n_states, 1), jnp.exp(params[idx]))
                idx += 1
            else:
                taus = jnp.exp(params[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            taus = jnp.array(init_taus)

        return log_startprob, log_mu, p_binom, alphas, taus

    @staticmethod
    def jax_dense_emissions(
        u_log_mu, u_alphas, u_p_binom, u_taus, X_nb, base_nb, X_bb, total_bb
    ):
        k_nb = X_nb[None, :, :]
        lam = base_nb[None, :, :] * jnp.exp(u_log_mu[:, None, :])
        r = 1.0 / jnp.maximum(u_alphas[:, None, :], 1e-10)
        p = 1.0 / (1.0 + u_alphas[:, None, :] * lam)

        log_emit_nb = jnp.where(
            lam <= 0.0,
            0.0,
            jsp.gammaln(k_nb + r)
            - jsp.gammaln(r)
            - jsp.gammaln(k_nb + 1)
            + r * jnp.log(p)
            + k_nb * jnp.log(1.0 - p),
        )

        k_bb = X_bb[None, :, :]
        n_bb = total_bb[None, :, :]
        alpha_bb = jnp.maximum(u_p_binom[:, None, :] * u_taus[:, None, :], 1e-10)
        beta_bb = jnp.maximum((1.0 - u_p_binom[:, None, :]) * u_taus[:, None, :], 1e-10)

        log_emit_bb = (
            jsp.gammaln(n_bb + 1)
            - jsp.gammaln(k_bb + 1)
            - jsp.gammaln(n_bb - k_bb + 1)
            + jsp.gammaln(k_bb + alpha_bb)
            + jsp.gammaln(n_bb - k_bb + beta_bb)
            - jsp.gammaln(n_bb + alpha_bb + beta_bb)
            - (
                jsp.gammaln(alpha_bb)
                + jsp.gammaln(beta_bb)
                - jsp.gammaln(alpha_bb + beta_bb)
            )
        )
        return log_emit_nb + log_emit_bb

    @staticmethod
    def jax_marginal_forward(
        log_startprob, log_transmat, log_emissions_sum, is_start, is_end
    ):
        def scan_step(prev_log_alpha, inputs):
            is_s, log_em = inputs
            trans_alpha = jsp.logsumexp(prev_log_alpha[:, None] + log_transmat, axis=0)
            next_alpha = jnp.where(is_s, log_startprob + log_em, trans_alpha + log_em)
            return next_alpha, next_alpha

        _, log_alpha_seq = jax.lax.scan(
            scan_step, jnp.zeros_like(log_startprob), (is_start, log_emissions_sum.T)
        )
        alpha_sums = jsp.logsumexp(log_alpha_seq.T, axis=0)

        total_nll = -jnp.sum(jnp.where(is_end, alpha_sums, 0.0))
        return total_nll

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

        # Re-use base class methods
        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

        logger.info(
            f"Encoders built. Medians: NB={np.median(nbEncoder.total_count):.4f} ({nbEncoder.compression_rate:.2%} comp), "
            f"BB={np.median(bbEncoder.total_count):.4f} ({bbEncoder.compression_rate:.2%} comp)."
        )

        log_mu, p_binom, alphas, taus, log_startprob, log_transmat = (
            self.get_initial_params(
                n_states, n_spots, init_log_mu, init_p_binom, init_alphas, init_taus
            )
        )

        logger.info(
            f"--- hmm initialized ({mode.upper()}) JAX backend ---\n"
            f"kwargs:\n{pprint.pformat(kwargs, indent=2) if kwargs else '{}'}\n"
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

        # Build JAX Arrays
        X_nb_jax = jnp.array(X[:, 0, :])
        X_bb_jax = jnp.array(X[:, 1, :])
        base_nb_jax = jnp.array(base_nb_mean)
        total_bb_jax = jnp.array(total_bb_RD)

        is_start_arr = np.zeros(X.shape[0], dtype=bool)
        is_end_arr = np.zeros(X.shape[0], dtype=bool)

        cumlen = 0
        for le in lengths:
            is_start_arr[cumlen] = True
            is_end_arr[cumlen + le - 1] = True
            cumlen += le

        is_start_jax = jnp.array(is_start_arr)
        is_end_jax = jnp.array(is_end_arr)
        log_transmat_jax = jnp.array(log_transmat)

        # Define specific JAX Cost Functions relying on static configurations
        def jax_em_cost(params, state_posteriors_jax):
            u_log_startprob, u_log_mu, u_p_binom, u_alphas, u_taus = self.jax_unpack(
                params,
                self.params,
                n_states,
                optimize_nb,
                fix_NB_dispersion,
                shared_NB_dispersion,
                fix_BB_dispersion,
                shared_BB_dispersion,
                use_logit,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
            )
            log_emissions = self.jax_dense_emissions(
                u_log_mu,
                u_alphas,
                u_p_binom,
                u_taus,
                X_nb_jax,
                base_nb_jax,
                X_bb_jax,
                total_bb_jax,
            )
            cost = -jnp.sum(state_posteriors_jax * log_emissions[..., 0])
            return cost, log_emissions

        def jax_marginal_cost(params):
            u_log_startprob, u_log_mu, u_p_binom, u_alphas, u_taus = self.jax_unpack(
                params,
                self.params,
                n_states,
                optimize_nb,
                fix_NB_dispersion,
                shared_NB_dispersion,
                fix_BB_dispersion,
                shared_BB_dispersion,
                use_logit,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
            )
            log_emissions = self.jax_dense_emissions(
                u_log_mu,
                u_alphas,
                u_p_binom,
                u_taus,
                X_nb_jax,
                base_nb_jax,
                X_bb_jax,
                total_bb_jax,
            )
            log_em_sum = jnp.sum(log_emissions, axis=-1)

            total_nll = self.jax_marginal_forward(
                u_log_startprob, log_transmat_jax, log_em_sum, is_start_jax, is_end_jax
            )
            return total_nll, log_emissions

        # Compile objective and gradient mappings
        if mode == "em":
            jax_val_and_grad = jax.jit(jax.value_and_grad(jax_em_cost, has_aux=True))
        else:
            jax_val_and_grad = jax.jit(
                jax.value_and_grad(jax_marginal_cost, has_aux=True)
            )

        # Setup scipy state management
        if mode == "em":
            self.log_emissions, self.state_posteriors = None, None
            self.log_startprob = log_startprob
            self.iterations = 0

            def callback(intermediate_result: OptimizeResult = None):
                if (self.iterations > 0) and (self.iterations % 2 != 0):
                    self.iterations += 1
                    return
                # Reuses the fast numba base class method for state posteriors
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

            # --- BOOTSTRAP THE FIRST E-STEP ---
            # Unpack x0 and compute the initial log_emissions manually
            # so the first callback() has data to calculate the state posteriors.
            u_log_sp, u_log_mu, u_p_binom, u_alphas, u_taus = self.jax_unpack(
                x0,
                self.params,
                n_states,
                optimize_nb,
                fix_NB_dispersion,
                shared_NB_dispersion,
                fix_BB_dispersion,
                shared_BB_dispersion,
                use_logit,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
            )
            init_log_em = self.jax_dense_emissions(
                u_log_mu,
                u_alphas,
                u_p_binom,
                u_taus,
                X_nb_jax,
                base_nb_jax,
                X_bb_jax,
                total_bb_jax,
            )
            self.log_emissions = np.array(init_log_em)
            callback(None)  # Initialize self.state_posteriors for iter 0

            def cost_fn(params):
                (val, log_em_jax), grad = jax_val_and_grad(
                    params, jnp.array(self.state_posteriors)
                )
                # Update emissions for the next callback (E-step)
                self.log_emissions = np.array(log_em_jax)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

        elif mode == "marginal":
            callback = None

            def cost_fn(params):
                (val, log_em_jax), grad = jax_val_and_grad(params)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

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
            f"Starting {mode} optimization with {optimizer} (JAX). Initial cost={cost_fn(x0)[0]:.6e}"
        )

        res = scipy.optimize.minimize(
            cost_fn,
            x0,
            method=optimizer,
            jac=True,
            bounds=bounds,
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

        # Fallback to base class numba loops to generate identical numpy outputs
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
