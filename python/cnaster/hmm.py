import logging

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from cnaster.hmm_initialize import gmm_init, cna_mixture_init
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmm_utils import compute_posterior_obs

logger = logging.getLogger(__name__)


def pipeline_baum_welch(
    output_prefix,
    X,
    lengths,
    n_states,
    base_nb_mean,
    total_bb_RD,
    log_sitewise_transmat,
    tumor_prop=None,
    hmmclass=hmm_sitewise,
    params="smp",
    t=1 - 1e-6,
    random_state=0,
    in_log_space=True,
    only_minor=False,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    **kwargs,
):
    logger.info(f"Solving HMM for X={X.shape} with {hmmclass.__name__} instance.")

    n_spots = X.shape[2]

    if ((init_log_mu is None) and ("m" in params)) or (
        (init_p_binom is None) and ("p" in params)
    ):
        cna_mixture_init():
            n_states,
            t,
            X,
            base_nb_mean,
            total_bb_RD,
            hmmclass,
        )

        exit(0)

        tmp_log_mu, tmp_p_binom = gmm_init(
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=in_log_space,
            only_minor=only_minor,
        )

        # TODO sort print
        if (init_log_mu is None) and ("m" in params):
            init_log_mu = tmp_log_mu
            logger.info(f"Initialized log_mu with GMM:\n{init_log_mu}")

        if (init_p_binom is None) and ("p" in params):
            init_p_binom = tmp_p_binom
            logger.info(f"Initialized p_binom with GMM:\n{init_p_binom}")
    else:
        if "m" in params:
            logger.info(f"Assumed initial log_mu:\n{init_log_mu}")
        if "p" in params:
            logger.info(f"Assumed initial p_binom:\n{init_p_binom}")

    hmm_model = hmmclass(params=params, t=t)

    remain_kwargs = {
        k: v for k, v in kwargs.items() if k in ["lambd", "sample_length", "log_gamma"]
    }

    (
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        new_log_startprob,
        new_log_transmat,
        log_gamma,
    ) = hmm_model.run_baum_welch_nb_bb(
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        tumor_prop,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        init_alphas=init_alphas,
        init_taus=init_taus,
        max_iter=max_iter,
        tol=tol,
        **remain_kwargs,
    )

    to_log = [
        f"Solved for best emission parameters with {hmm_model.__class__.__name__}:"
    ]

    if "m" in params and new_log_mu is not None:
        to_log.append(f"log_mu=\n{new_log_mu}")

    if "p" in params and new_p_binom is not None:
        to_log.append(f"p_binom=\n{new_p_binom}")

    logger.info("\n".join(to_log))

    if tumor_prop is None:
        (
            log_emission_rdr,
            log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, new_log_mu, new_alphas, total_bb_RD, new_p_binom, new_taus
        )
    else:
        if ("m" in params) and ("sample_length" in kwargs):
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
                        new_log_mu[this_pred_cnv, :]
                        + np.log(kwargs["lambd"]).reshape(-1, 1),
                        axis=0,
                    )
                )

            logmu_shift = np.vstack(logmu_shift)

            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                new_log_mu,
                new_alphas,
                total_bb_RD,
                new_p_binom,
                new_taus,
                tumor_prop,
                logmu_shift=logmu_shift,
                sample_length=kwargs["sample_length"],
            )
        else:
            (
                log_emission_rdr,
                log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                new_log_mu,
                new_alphas,
                total_bb_RD,
                new_p_binom,
                new_taus,
                tumor_prop,
            )

    log_emission = log_emission_rdr + log_emission_baf

    log_alpha = hmmclass.forward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )

    llf = np.sum(scipy.special.logsumexp(log_alpha[:, np.cumsum(lengths) - 1], axis=0))

    log_beta = hmmclass.backward_lattice(
        lengths,
        new_log_transmat,
        new_log_startprob,
        log_emission,
        log_sitewise_transmat,
    )

    # NB compute state posterior.
    log_gamma = compute_posterior_obs(log_alpha, log_beta)

    # NB with phasing.
    pred = np.argmax(log_gamma, axis=0)

    # NB copy number only.
    pred_cnv = pred % n_states

    """
    if output_prefix is not None:
        tmp = np.log10(1. - t)

        np.savez(
            f"{output_prefix}_nstates{n_states}_{params}_{tmp:.0f}_seed{random_state}.npz",
            new_log_mu=new_log_mu,
            new_alphas=new_alphas,
            new_p_binom=new_p_binom,
            new_taus=new_taus,
            new_log_startprob=new_log_startprob,
            new_log_transmat=new_log_transmat,
            log_gamma=log_gamma,
            pred_cnv=pred_cnv,
            llf=llf,
        )
    """

    return {
        "new_log_mu": new_log_mu,
        "new_alphas": new_alphas,
        "new_p_binom": new_p_binom,
        "new_taus": new_taus,
        "new_log_startprob": new_log_startprob,
        "new_log_transmat": new_log_transmat,
        "log_gamma": log_gamma,
        "pred_cnv": pred_cnv,
        "llf": llf,
    }
