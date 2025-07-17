import logging

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from sklearn.mixture import GaussianMixture
from cnaster.hmm_utils import compute_posterior_obs
from cnaster.hmm_sitewise import hmm_sitewise

logger = logging.getLogger(__name__)


def initialization_by_gmm(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    min_binom_prob=0.1,
    max_binom_prob=0.9,
):
    X_gmm_rdr, X_gmm_baf = None, None

    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            offset = np.mean(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            ) - np.min(X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))])
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            offset = 0
            normalizetomax1 = np.max(
                X_gmm_rdr[(~np.isnan(X_gmm_rdr)) & (~np.isinf(X_gmm_rdr))]
            )
            X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1

    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T
        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob

    # combine RDR and BAF
    if ("m" in params) and ("p" in params):
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        X_gmm = X_gmm_baf

    # deal with NAN
    for k in range(X_gmm.shape[1]):
        last_idx_notna = -1
        for i in range(X_gmm.shape[0]):
            if last_idx_notna >= 0 and np.isnan(X_gmm[i, k]):
                X_gmm[i, k] = X_gmm[last_idx_notna, k]
            elif not np.isnan(X_gmm[i, k]):
                last_idx_notna = i

    X_gmm = X_gmm[np.sum(np.isnan(X_gmm), axis=1) == 0, :]

    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=1).fit(X_gmm)
    else:
        gmm = GaussianMixture(
            n_components=n_states, max_iter=1, random_state=random_state
        ).fit(X_gmm)

    # turn gmm fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = (
            gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = gmm.means_[:, X.shape[2] :]
        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1 - gmm_p_binom, gmm_p_binom)
    elif "m" in params:
        gmm_log_mu = (
            gmm.means_ * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = None
    elif "p" in params:
        gmm_log_mu = None
        gmm_p_binom = gmm.means_

        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom)

    return gmm_log_mu, gmm_p_binom


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
    n_spots = X.shape[2]

    if ((init_log_mu is None) and ("m" in params)) or (
        (init_p_binom is None) and ("p" in params)
    ):
        logger.info(f"Running Gaussian mixture model for initialization.")

        tmp_log_mu, tmp_p_binom = initialization_by_gmm(
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=in_log_space,
            only_minor=only_minor,
        )

        if (init_log_mu is None) and ("m" in params):
            init_log_mu = tmp_log_mu

        if (init_p_binom is None) and ("p" in params):
            init_p_binom = tmp_p_binom

    logger.info(f"Initialized log_mu:\n{init_log_mu}")
    logger.info(f"Initialized p_binom:\n{init_p_binom}")

    logger.info(f"Solving HMM with {hmm_sitewise.__name__} instance.")

    hmmmodel = hmmclass(params=params, t=t)

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
    ) = hmmmodel.run_baum_welch_nb_bb(
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

    logger.info(
        f"Solved for best state configuration:\nmu={np.exp(new_log_mu)}\nb={np.exp(new_p_binom)}"
    )

    # likelihood
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
    if not output_prefix is None:
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
