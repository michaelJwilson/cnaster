import time
import numpy as np
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax.scipy.special import logsumexp, expit, gammaln, betaln
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def run_marginal_likelihood_nb_bb_jax(
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
    """
    JAX-accelerated direct marginal likelihood maximization.
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

    # Note: Encoders retained for final posterior computation, but JAX bypasses them for raw speed
    # nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
    # bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

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

    logger.info("Compiling JAX exact Marginal Likelihood objective and gradients...")

    jax_X_rdr = jnp.array(X[:, 0, 0])
    jax_base_nb_mean = jnp.array(base_nb_mean[:, 0])
    jax_X_baf = jnp.array(X[:, 1, 0])
    jax_total_bb_RD = jnp.array(total_bb_RD[:, 0])

    jax_log_transmat = jnp.array(log_transmat)

    def jax_unpack(flat_params):
        idx = 0
        if "s" in self.params:
            raw_start = flat_params[idx : idx + n_states]
            j_log_startprob = raw_start - logsumexp(raw_start)
            idx += n_states
        else:
            j_log_startprob = jnp.array(log_startprob)

        if optimize_nb and "m" in self.params:
            j_log_mu = flat_params[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            j_log_mu = jnp.array(log_mu)

        if "p" in self.params:
            if use_logit:
                j_p_binom = expit(
                    flat_params[idx : idx + n_states].reshape(n_states, 1)
                )
            else:
                j_p_binom = jnp.clip(
                    flat_params[idx : idx + n_states].reshape(n_states, 1),
                    1e-6,
                    1 - 1e-6,
                )
            idx += n_states
        else:
            j_p_binom = jnp.array(p_binom)

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                j_alphas = jnp.full((n_states, 1), jnp.exp(flat_params[idx]))
                idx += 1
            else:
                j_alphas = jnp.exp(flat_params[idx : idx + n_states]).reshape(
                    n_states, 1
                )
                idx += n_states
        else:
            j_alphas = jnp.array(alphas)

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                j_taus = jnp.full((n_states, 1), jnp.exp(flat_params[idx]))
                idx += 1
            else:
                j_taus = jnp.exp(flat_params[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            j_taus = jnp.array(taus)

        return j_log_startprob, j_log_mu, j_p_binom, j_alphas, j_taus

    def jax_forward_contig(start_p, emissions):
        def scan_fn(prev_alpha, curr_emission):
            next_alpha = (
                logsumexp(prev_alpha[:, None] + jax_log_transmat, axis=0)
                + curr_emission
            )
            return next_alpha, next_alpha

        init_alpha = start_p + emissions[:, 0]
        final_alpha, _ = jax.lax.scan(scan_fn, init_alpha, emissions[:, 1:].T)
        return logsumexp(final_alpha)

    @jax.jit
    def jax_nll_objective(flat_params):
        j_log_start, j_log_mu, j_p_binom, j_alphas, j_taus = jax_unpack(flat_params)

        mu_nb = jax_base_nb_mean * jnp.exp(j_log_mu)
        n_nb = 1.0 / j_alphas
        p_nb = n_nb / (n_nb + mu_nb)
        log_rdr = jstats.nbinom.logpmf(jax_X_rdr, n_nb, p_nb)
        log_rdr = jnp.where(jax_base_nb_mean > 0, log_rdr, 0.0)

        a = j_p_binom * j_taus
        b = (1.0 - j_p_binom) * j_taus
        log_comb = (
            gammaln(jax_total_bb_RD + 1)
            - gammaln(jax_X_baf + 1)
            - gammaln(jax_total_bb_RD - jax_X_baf + 1)
        )
        log_baf = (
            log_comb
            + betaln(jax_X_baf + a, jax_total_bb_RD - jax_X_baf + b)
            - betaln(a, b)
        )
        log_baf = jnp.where(jax_total_bb_RD > 0, log_baf, 0.0)

        log_emissions = log_rdr + log_baf

        total_nll = 0.0
        curr = 0
        for le in lengths:
            contig_emissions = log_emissions[:, curr : curr + le]
            total_nll += -jax_forward_contig(j_log_start, contig_emissions)
            curr += le

        return total_nll

    value_and_grad_fn = jax.value_and_grad(jax_nll_objective)

    def objective_fn(params_np):
        v, g = value_and_grad_fn(params_np)
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

    runtime = time.time() - start_time_opt
    logger.info(
        f"JAX Optimization complete: {runtime:.2f}s\n"
        f"Iter: {res.nit}, Evals: {res.nfev}\n"
        f"Converged: {res.success} ({res.message})\n"
        f"NLL: {res.fun:.6e}"
    )

    # ---------------------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------------------
    if propagate_errors:
        log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err = (
            self.unpack_param_errors(
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

    log_emission_rdr, log_emission_baf = self.compute_emission_probability_nb_betabinom(
        X,
        base_nb_mean,
        final_log_mu,
        final_alphas,
        total_bb_RD,
        final_p_binom,
        final_taus,
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


class hmm_nophasing_jax(hmm_nophasing):
    # TODO
    run_baum_welch_nb_bb = run_marginal_likelihood_nb_bb_jax
