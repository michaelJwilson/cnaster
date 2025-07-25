import logging

import numpy as np
from cnaster.config import get_global_config
from sklearn.mixture import GaussianMixture

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
    logger.info(f"Initializing HMM emission with Gaussian Mixture Model assuming only_minor={only_minor}.")

    X_gmm_rdr, X_gmm_baf = None, None

    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            offset = np.mean(X_gmm_rdr[valid])
            normalizetomax1 = np.max(X_gmm_rdr[valid]) - np.min(X_gmm_rdr[valid])

            logger.info(f"Assuming log-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}")
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            offset = 0

            # NB TODO? assumes X_gmm_rdr.min() = 0.
            normalizetomax1 = np.max(X_gmm_rdr[valid])

            logger.info(f"Assuming linear-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}") 
                   
        X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1

    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T

        clipped = (X_gmm_baf < min_binom_prob) | (X_gmm_baf > max_binom_prob)

        logger.warning(f"Clipping {np.mean(clipped):.4f} of BAF values to [{min_binom_prob}, {max_binom_prob}].")

        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob

    if ("m" in params) and ("p" in params):
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        X_gmm = X_gmm_baf

    # NB resolve NAN
    num_patched = 0

    for k in range(X_gmm.shape[1]):
        last_idx_notna = -1
        for i in range(X_gmm.shape[0]):
            if last_idx_notna >= 0 and np.isnan(X_gmm[i, k]):
                X_gmm[i, k] = X_gmm[last_idx_notna, k]
                num_patched += 1
            elif not np.isnan(X_gmm[i, k]):
                last_idx_notna = i

    logger.info(f"Patched {num_patched/X_gmm.shape[1]:.4f} values with NaNs in input data.")

    valid = np.sum(np.isnan(X_gmm), axis=1) == 0
    X_gmm = X_gmm[valid, :]

    logger.info(f"Retained {np.mean(valid)} of samples after patching.")

    max_iter = get_global_config().hmm.gmm_maxiter

    # DEPRECATE if/else.
    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=max_iter).fit(X_gmm)
    else:
        gmm = GaussianMixture(
            n_components=n_states, max_iter=max_iter, random_state=random_state
        ).fit(X_gmm)

    # TODO check? score() returns per-sample log-likelihood
    logger.info(
        f"GMM: score={gmm.score(X_gmm):.6f}, converged={gmm.converged_}, iterations={gmm.n_iter_}"
    )

    # NB cast GMM fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = (
            gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = gmm.means_[:, X.shape[2] :]

        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1. - gmm_p_binom, gmm_p_binom)

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
