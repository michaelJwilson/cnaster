import logging

import numpy as np
from numba import njit
from pysam import samples
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from cnaster.config import get_global_config
from cnaster.hmm_emission import (
    Weighted_BetaBinom_mix,
    Weighted_NegativeBinomial_mix,
    nloglikeobs_bb,
)
from cnaster.hmm_update import get_em_solver_params
from cnaster.utils import top_hat_sum, cast_clone_label
from cnaster.config import get_global_config
from cnaster.utils import write_fig
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmm_nophasing import hmm_nophasing
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def get_eff_element(t, K, two_sided=True):
    result = -np.log((t * K - 1.0) / (K - 1.0))
    result = 1.0 / result

    if two_sided:
        result *= 2.0

    return int(np.ceil(result))


@njit
def interval_mean(arr, N):
    num_groups = arr.shape[0] // N

    # TODO generalize shape definition
    result = np.empty((num_groups, *arr.shape[1:]), dtype=arr.dtype)

    for j in range(num_groups):
        start_idx = j * N
        end_idx = start_idx + N
        result[j, ...] = np.mean(arr[start_idx:end_idx, ...], axis=0)

    return result


def cna_mixture_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    max_iter=15,
    width=None,
):
    logger.info(f"Initializing HMM emission with CNA Mixture++.")

    known_normal = np.any(base_nb_mean)
    num_segments, _, num_spots = X.shape

    if width is not None:
        X = top_hat_sum(X, width)
        base_nb_mean = top_hat_sum(base_nb_mean, width)
        total_bb_RD = top_hat_sum(total_bb_RD, width)

    solution, solution_lnlike = None, -np.inf

    num_to_solve = 10 * 10 * max_iter
    num_solved = 0

    if known_normal:
        grid_alphas = np.logspace(-3, -1, num=10, base=10.0)
    else:
        grid_alphas = np.logspace(-3, -1, num=1, base=10.0)

    grid_taus = np.arange(10, 1_011, 100)
        
    # TODO HACK?
    for alpha in grid_alphas:
        for tau in grid_taus:
            alphas = alpha * np.ones((n_states, 1))
            taus = tau * np.ones((n_states, 1))

            for ii in range(max_iter):
                log_mu, p_binom = np.array([0.0]).reshape((1, 1)), np.array(
                    [0.5]
                ).reshape((1, 1))

                while len(log_mu) < n_states:
                    # NB (n_states, n_obs, n_spots)
                    lnlike_rdr, lnlike_baf = (
                        hmm_sitewise.compute_emission_probability_nb_betabinom(
                            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                        )
                    )

                    # NB lnlike_rdr is zero (null op. for additiona) until normal spots are defined.
                    lnlike = lnlike_rdr + lnlike_baf

                    # TODO track finite.
                    lnlike[~np.isfinite(lnlike)] = -np.inf

                    best_lnlike = np.max(lnlike, axis=0)

                    total_best_lnlike = best_lnlike.sum()

                    ps = -best_lnlike.ravel()
                    ps /= ps.sum()

                    try:
                        sample_idx = np.random.choice(
                            np.arange(num_segments * num_spots), p=ps
                        )
                    except:
                        continue

                    sample_segment, sample_spot = (
                        sample_idx // num_spots,
                        sample_idx % num_spots,
                    )

                    sample_ln_rdr = np.log(
                        X[sample_segment, 0, sample_spot]
                        / base_nb_mean[sample_segment, 0]
                    )
                    sample_baf = (
                        X[sample_segment, 1, sample_spot]
                        / total_bb_RD[sample_segment, 0]
                    )

                    # TODO
                    if known_normal and (
                        ~np.isfinite(sample_ln_rdr) or ~np.isfinite(sample_baf)
                    ):
                        continue

                    log_mu = np.vstack([log_mu, [[sample_ln_rdr]]])
                    p_binom = np.vstack([p_binom, [[sample_baf]]])

                    if total_best_lnlike > solution_lnlike:
                        solution = [log_mu, alphas, p_binom, taus]
                        solution_lnlike = total_best_lnlike

                        logger.info(
                            f"Found new best initialization ({num_solved}/{num_to_solve}) with copy mixture++ and lnlike={solution_lnlike:.6e}:\nlog_mu={log_mu},\nalphas={alphas},\np_binom={p_binom},\ntaus={taus}."
                        )

                logger.debug(alpha, tau, ii, total_best_lnlike, p_binom.tolist())

                num_solved += 1

    log_mu, alphas, p_binom, taus = solution

    if not known_normal:
        log_mu, alphas = None, None

    logger.info(
        f"Solved for initial parameters with  copy mixture++ and lnlike={solution_lnlike:.6e}:\nlog_mu={log_mu},\nalphas={alphas},\np_binom={p_binom},\ntaus={taus}."
    )

    return log_mu, alphas, p_binom, taus


# TODO define width
def plot_cna_mixture(
    init_log_mu, init_p_binom, X, base_nb_mean, total_bb_RD, width=1, prefix="initial"
):
    logger.info(f"Plotting initial copy state mixture for X.shape={X.shape}.")

    # NB base_nb_mean is zero until post-BAF normal identication; in which case,
    #    these will be NAN.
    X_gmm_rdr = np.vstack(
        [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
    ).T

    assert X_gmm_rdr.shape == (len(X[:, 0, 0]), X.shape[2])

    valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

    if np.all(~valid):
        X_gmm_rdr[~valid] = np.random.normal(
            loc=1.0, scale=0.25, size=np.count_nonzero(~valid)
        )

    # TODO clipping?
    X_gmm_baf = np.vstack(
        [
            top_hat_sum(X[:, 1, s], width) / top_hat_sum(total_bb_RD[:, s], width)
            for s in range(X.shape[2])
        ]
    ).T

    if init_log_mu is not None:
        init_mu = np.exp(init_log_mu)
    else:
        init_mu = np.ones_like(init_p_binom)

    num_clones, num_segments = X.shape[2], X.shape[0]

    # NB S,n = 3,4 ... [0 1 2 0 1 2 0 1 2 0 1 2], i.e. column major.
    clone_idx = np.tile(np.arange(num_clones), num_segments)

    palette = sns.color_palette(n_colors=num_clones)

    x = X_gmm_baf.ravel()
    y = X_gmm_rdr.ravel()

    g = sns.JointGrid(x=x, y=y, height=8, ratio=3, space=0.15)
    valid_mask = np.isfinite(x) & np.isfinite(y)

    for c in range(num_clones):
        clone_mask = (clone_idx == c) & valid_mask

        if np.any(clone_mask):
            g.ax_joint.scatter(
                x[clone_mask],
                y[clone_mask],
                s=1,
                marker=".",
                alpha=0.6,
                color=palette[c],
            )

    g.ax_joint.scatter(
        init_p_binom, init_mu, marker="*", facecolor="none", edgecolor="k", s=25
    )

    bins = 50

    validx = np.isfinite(x)
    validy = np.isfinite(y)

    # bins_x = np.histogram_bin_edges(x[validx], bins=bins)
    # bins_y = np.histogram_bin_edges(y[validy], bins=bins)

    bins_x = np.arange(-0.01, 0.6, 5.0e-3)
    bins_y = np.arange(-0.1, 10.0, 0.1)

    centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    width_x = bins_x[1] - bins_x[0]

    centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    height_y = bins_y[1] - bins_y[0]

    legend_patches = []

    for c in range(num_clones):
        clone_mask = clone_idx == c

        assert np.any(clone_mask)

        counts_x, _ = np.histogram(x[clone_mask], bins=bins_x)
        counts_y, _ = np.histogram(y[clone_mask], bins=bins_y)

        g.ax_marg_x.bar(
            centers_x,
            counts_x,
            width=width_x,
            align="center",
            facecolor="none",
            edgecolor=palette[c],
            linewidth=1.0,
            alpha=1.0,
        )

        g.ax_marg_y.barh(
            centers_y,
            counts_y,
            height=height_y,
            align="center",
            facecolor="none",
            edgecolor=palette[c],
            linewidth=1.0,
            alpha=0.5,
        )

        legend_patches.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor=palette[c],
                label=cast_clone_label(f"clone {c}"),
            )
        )

    g.set_axis_labels("ZHF", "RDR")
    g.ax_joint.legend(handles=legend_patches, loc="upper left", framealpha=0.0)

    fig = g.fig

    config = get_global_config()
    fig_path = f"{config.paths.output_dir}/plots/{prefix}_rdr_baf.pdf"

    logger.info(f"Writing initial copy state mixture plot to {fig_path}")

    write_fig(fig_path, fig, transparent=True, bbox_inches="tight")


def gmm_init(
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
    logger.info(
        f"Initializing HMM emission with Gaussian Mixture Model assuming only_minor={only_minor}."
    )

    X_gmm_rdr, X_gmm_baf = None, None

    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            if not np.any(valid):
                logger.error(
                    f"No valid RDR data given sum(base_nb_mean)={sum(base_nb_mean)}"
                )
                raise RuntimeError()

            offset = np.mean(X_gmm_rdr[valid])
            normalizetomax1 = np.max(X_gmm_rdr[valid]) - np.min(X_gmm_rdr[valid])

            logger.info(
                f"Assuming log-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}"
            )
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            if not np.any(valid):
                logger.error(
                    f"No valid RDR data given sum(base_nb_mean)={sum(base_nb_mean)}"
                )
                raise RuntimeError()

            offset = 0
            normalizetomax1 = np.max(X_gmm_rdr[valid])

            logger.info(
                f"Assuming linear-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}"
            )

        X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1

    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T

        clipped = (X_gmm_baf < min_binom_prob) | (X_gmm_baf > max_binom_prob)

        logger.warning(
            f"Clipping {np.mean(clipped):.4f} of BAF values to [{min_binom_prob}, {max_binom_prob}]."
        )

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

    logger.info(
        f"Patched {num_patched/X_gmm.shape[1]:.4f} values with NaNs in input data."
    )

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
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom)

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

    if np.any(gmm_p_binom > 0.5):
        logger.warning(
            f"GMM initialized p binom > 0.5, {gmm_p_binom[gmm_p_binom > 0.5]}"
        )

    logger.info(f"Solved for GMM initialized parameters:\n{gmm_log_mu}\n{gmm_p_binom}")

    return gmm_log_mu, gmm_p_binom
