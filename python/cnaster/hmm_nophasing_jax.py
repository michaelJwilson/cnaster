import os

# NB: These MUST be set before jax or any cnaster modules are imported
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import functools
import time
import numpy as np
import scipy.optimize

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax.scipy.special import logsumexp, expit, gammaln, betaln
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# Enable 64-bit precision to maintain gradient flow on long contigs
jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------------
# PURE JAX FUNCTIONS (Defined outside the class to allow JIT caching)
# -------------------------------------------------------------------------

def jax_unpack(
    flat_params,
    n_states,
    params_str,
    optimize_nb,
    fix_NB,
    shared_NB,
    fix_BB,
    shared_BB,
    use_logit,
    log_start_init,
    log_mu_init,
    p_binom_init,
    alphas_init,
    taus_init,
):
    """Pure JAX parameter unpacking for the objective function."""
    idx = 0

    if "s" in params_str:
        raw_start = flat_params[idx : idx + n_states]
        j_log_startprob = raw_start - logsumexp(raw_start)
        idx += n_states
    else:
        j_log_startprob = jnp.array(log_start_init)

    if optimize_nb and "m" in params_str:
        j_log_mu = flat_params[idx : idx + n_states].reshape(n_states, 1)
        idx += n_states
    else:
        j_log_mu = jnp.array(log_mu_init)

    if "p" in params_str:
        if use_logit:
            j_p_binom = expit(flat_params[idx : idx + n_states].reshape(n_states, 1))
        else:
            # Note: Preferred approach is using use_logit=True to map coordinates smoothly
            j_p_binom = jnp.clip(
                flat_params[idx : idx + n_states].reshape(n_states, 1), 1e-6, 1 - 1e-6
            )
        idx += n_states
    else:
        j_p_binom = jnp.array(p_binom_init)

    if optimize_nb and "m" in params_str and not fix_NB:
        if shared_NB:
            j_alphas = jnp.full((n_states, 1), jnp.exp(flat_params[idx]))
            idx += 1
        else:
            j_alphas = jnp.exp(flat_params[idx : idx + n_states]).reshape(n_states, 1)
            idx += n_states
    else:
        j_alphas = jnp.array(alphas_init)

    if "p" in params_str and not fix_BB:
        if shared_BB:
            j_taus = jnp.full((n_states, 1), jnp.exp(flat_params[idx]))
            idx += 1
        else:
            j_taus = jnp.exp(flat_params[idx : idx + n_states]).reshape(n_states, 1)
            idx += n_states
    else:
        j_taus = jnp.array(taus_init)

    return j_log_startprob, j_log_mu, j_p_binom, j_alphas, j_taus

# NB compiles once per set of static arguments.
@functools.partial(
    jax.jit,
    static_argnames=[
        "n_states",
        "params_str",
        "optimize_nb",
        "fix_NB",
        "shared_NB",
        "fix_BB",
        "shared_BB",
        "use_logit",
        "lengths",
    ],
)
def jax_nll_objective(
    flat_params,
    X_rdr,
    base_nb_mean,
    X_baf,
    total_bb_RD,
    log_transmat,
    log_start_init,
    log_mu_init,
    p_binom_init,
    alphas_init,
    taus_init,
    n_states,
    params_str,
    optimize_nb,
    fix_NB,
    shared_NB,
    fix_BB,
    shared_BB,
    use_logit,
    lengths,
):
    j_log_start, j_log_mu, j_p_binom, j_alphas, j_taus = jax_unpack(
        flat_params,
        n_states,
        params_str,
        optimize_nb,
        fix_NB,
        shared_NB,
        fix_BB,
        shared_BB,
        use_logit,
        log_start_init,
        log_mu_init,
        p_binom_init,
        alphas_init,
        taus_init,
    )
    mu_nb = base_nb_mean * jnp.exp(j_log_mu)
    n_nb = 1.0 / j_alphas
    p_nb = n_nb / (n_nb + mu_nb)
    log_rdr = jnp.where(base_nb_mean > 0, jstats.nbinom.logpmf(X_rdr, n_nb, p_nb), 0.0)

    a = j_p_binom * j_taus
    b = (1.0 - j_p_binom) * j_taus
    log_comb = (
        gammaln(total_bb_RD + 1) - gammaln(X_baf + 1) - gammaln(total_bb_RD - X_baf + 1)
    )
    log_baf = jnp.where(
        total_bb_RD > 0,
        log_comb + betaln(X_baf + a, total_bb_RD - X_baf + b) - betaln(a, b),
        0.0,
    )

    # NB mitigate underflow variance poisoning by clamping lowest possible log likelihoods
    MIN_LOG_PROB = -1e4
    log_emissions = jnp.maximum(log_rdr, MIN_LOG_PROB) + jnp.maximum(log_baf, MIN_LOG_PROB)

    # NB forward algorithm (scan).
    def scan_fn(prev_alpha, curr_emission):
        next_alpha = (
            logsumexp(prev_alpha[:, None] + log_transmat, axis=0) + curr_emission
        )
        return next_alpha

    total_nll = 0.0
    curr = 0
    for le in lengths:
        contig_emissions = log_emissions[:, curr : curr + le]
        init_alpha = j_log_start + contig_emissions[:, 0]
        final_alpha = jax.lax.scan(scan_fn, init_alpha, contig_emissions[:, 1:].T)
        total_nll += -logsumexp(final_alpha)
        curr += le

    return total_nll


# NB automatically generate optimized gradient and hessian functions based on the JIT objective
jax_value_and_grad = jax.jit(
    jax.value_and_grad(jax_nll_objective, argnums=0),
    static_argnames=[
        "n_states",
        "params_str",
        "optimize_nb",
        "fix_NB",
        "shared_NB",
        "fix_BB",
        "shared_BB",
        "use_logit",
        "lengths",
    ],
)
jax_exact_hessian = jax.jit(
    jax.hessian(jax_nll_objective, argnums=0),
    static_argnames=[
        "n_states",
        "params_str",
        "optimize_nb",
        "fix_NB",
        "shared_NB",
        "fix_BB",
        "shared_BB",
        "use_logit",
        "lengths",
    ],
)

class hmm_nophasing_jax(hmm_nophasing):
    def unpack_param_errors(
        self,
        x,
        exact_cov_matrix,
        n_states,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        """
        Vectorized error unpacking utilizing the jax inverse Hessian.
        Applies the Delta method for parameter transformations.
        """
        idx = 0

        # NB base standard errors (variance diagonal)
        parameter_errors_diag = np.sqrt(
            np.clip(np.diag(exact_cov_matrix), a_min=0, a_max=None)
        )

        # 1. Start Probabilities (Softmax Delta Method)
        if "s" in self.params:
            raw_startprob = x[idx : idx + n_states]
            cov_raw = exact_cov_matrix[idx : idx + n_states, idx : idx + n_states]

            p_start = scipy.special.softmax(raw_startprob)
            J = np.diag(p_start) - np.outer(p_start, p_start)

            cov_transformed = J @ cov_raw @ J.T
            log_startprob_err = np.sqrt(
                np.clip(np.diag(cov_transformed), a_min=0, a_max=None)
            )
            idx += n_states
        else:
            log_startprob_err = None

        # 2. RDR Mean (Linear scale parameter, direct extraction)
        if optimize_nb and "m" in self.params:
            log_mu_err = parameter_errors_diag[idx : idx + n_states].reshape(
                n_states, 1
            )
            idx += n_states
        else:
            log_mu_err = None

        # 3. BAF Binomial Probability (Logit Delta Method)
        if "p" in self.params:
            raw_p_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
            if use_logit:
                p_binom_val = scipy.special.expit(
                    x[idx : idx + n_states].reshape(n_states, 1)
                )
                # Delta Method: Var(f(x)) = [f'(x)]^2 * Var(x) -> SE = |f'(x)| * SE_x
                p_binom_err = p_binom_val * (1.0 - p_binom_val) * raw_p_err
            else:
                p_binom_err = raw_p_err
            idx += n_states
        else:
            p_binom_err = None

        # 4. NB Dispersion (Exponential Delta Method: alpha = exp(raw_alpha))
        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                alpha_err = np.exp(x[idx]) * parameter_errors_diag[idx]
                alphas_err = np.full((n_states, 1), alpha_err)
                idx += 1
            else:
                raw_alpha_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )
                alphas_err = (
                    np.exp(x[idx : idx + n_states].reshape(n_states, 1)) * raw_alpha_err
                )
                idx += n_states
        else:
            alphas_err = None

        # 5. BB Dispersion (Exponential Delta Method: tau = exp(raw_tau))
        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                tau_err = np.exp(x[idx]) * parameter_errors_diag[idx]
                taus_err = np.full((n_states, 1), tau_err)
                idx += 1
            else:
                raw_tau_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )
                taus_err = (
                    np.exp(x[idx : idx + n_states].reshape(n_states, 1)) * raw_tau_err
                )
                idx += n_states
        else:
            taus_err = None

        return log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err

    def run_marginal_likelihood_nb_bb(
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
        use_logit=True,
        propagate_errors=False,
        **kwargs,
    ):
        """
        JAX-accelerated direct marginal likelihood maximization with exact Hessian error propagation.
        """
        _, n_comp, n_spots = X.shape
        assert n_spots == 1, "Only single-spot processing supported in this block."
        assert n_comp == 2

        base_nb_mean = base_nb_mean.copy()

        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0
                base_nb_mean[est_rdr > max_rdr] = 0.0

        optimize_nb = np.any(base_nb_mean > 0)

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

        jax_X_rdr = jnp.array(X[:, 0, 0])
        jax_base_nb_mean = jnp.array(base_nb_mean[:, 0])
        jax_X_baf = jnp.array(X[:, 1, 0])
        jax_total_bb_RD = jnp.array(total_bb_RD[:, 0])
        jax_log_transmat = jnp.array(log_transmat)

        # NB pass lengths as a tuple so JAX can hash it as a static compile-time argument
        static_lengths = tuple(lengths)

        # TODO jax optimization.
        def objective_fn(params_np):
            v, g = jax_value_and_grad(
                jnp.array(params_np),
                jax_X_rdr,
                jax_base_nb_mean,
                jax_X_baf,
                jax_total_bb_RD,
                jax_log_transmat,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                n_states,
                self.params,
                optimize_nb,
                fix_NB_dispersion,
                shared_NB_dispersion,
                fix_BB_dispersion,
                shared_BB_dispersion,
                use_logit,
                static_lengths,
            )
            return np.array(v, dtype=np.float64), np.array(g, dtype=np.float64)

        start_time_opt = time.time()
        logger.info(f"Starting JAX L-BFGS-B Optimization with {len(x0)} parameters.")

        res = scipy.optimize.minimize(
            objective_fn,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": kwargs.get("max_iter", max_iter), "disp": False},
        )

        logger.info(
            f"JAX optimization complete: {time.time() - start_time_opt:.2f}s. NLL: {res.fun:.6e}"
        )

        if propagate_errors:
            logger.info(
                "Computing exact Hessian via JAX for parameter standard errors..."
            )

            # TODO direct jax inverse hessian? 
            H_jax = jax_exact_hessian(
                jnp.array(res.x),
                jax_X_rdr,
                jax_base_nb_mean,
                jax_X_baf,
                jax_total_bb_RD,
                jax_log_transmat,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                n_states,
                self.params,
                optimize_nb,
                fix_NB_dispersion,
                shared_NB_dispersion,
                fix_BB_dispersion,
                shared_BB_dispersion,
                use_logit,
                static_lengths,
            )
            H_np = np.array(H_jax, dtype=np.float64)

            try:
                exact_cov_matrix = np.linalg.inv(H_np)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Exact Hessian is strictly singular! Applying pseudo-inverse. Interpret standard errors carefully."
                )
                exact_cov_matrix = np.linalg.pinv(H_np)

            # Route to our overridden Delta method
            log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err = (
                self.unpack_param_errors(
                    x=res.x,
                    exact_cov_matrix=exact_cov_matrix,
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

        return {
            "new_log_mu": final_log_mu,
            "new_alphas": final_alphas,
            "new_p_binom": final_p_binom,
            "new_taus": final_taus,
            "new_log_startprob": final_log_startprob,
            "new_log_transmat": log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -float(res.fun),
            "n_states": n_states,
        } | param_errors