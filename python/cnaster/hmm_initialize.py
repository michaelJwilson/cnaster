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
    get_nbinom_start_params,
    get_betabinom_start_params,
)
from cnaster.hmm_update import get_em_solver_params
from cnaster.utils import top_hat_sum, cast_clone_label
from cnaster.config import get_global_config
from cnaster.utils import write_fig
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmm_nophasing import hmm_nophasing
from joblib import Parallel, delayed
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


def cna_mixture_init_search(
    alpha,
    tau,
    max_iter,
    anneal,
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    max_rdr,
    known_normal,
):
    alphas = alpha * np.ones((n_states, 1))
    taus = tau * np.ones((n_states, 1))

    num_segments, _, num_spots = X.shape

    # NB pre-compute flat sampling probabilities (uniform initially)
    flat_idx = np.arange(num_segments * num_spots)
    solution, solution_lnlike = None, -np.inf

    inital_log_mu, initial_p_binom = np.array([0.0 if known_normal else np.nan]).reshape(
        (1, 1)
    ), np.array([0.5]).reshape((1, 1))

    initial_lnlike_rdr, initial_lnlike_baf = (
        hmm_sitewise.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, inital_log_mu, alphas, total_bb_RD, initial_p_binom, taus
        )
    )

    for _ in range(max_iter):
        log_mu, p_binom = inital_log_mu.copy(), initial_p_binom.copy()

        while len(log_mu) < n_states:
            # NB (n_states, n_obs, n_spots) where n_states includes phase flip complement.
            if len(log_mu) == 1:
                lnlike_rdr, lnlike_baf = initial_lnlike_rdr, initial_lnlike_baf
            else:
                lnlike_rdr, lnlike_baf = (
                    hmm_sitewise.compute_emission_probability_nb_betabinom(
                        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
                    )
                )

            # NB lnlike_rdr is zero (null op. for additiona) until normal spots are defined.
            lnlike = lnlike_rdr + lnlike_baf

            # TODO track finite.
            lnlike[~np.isfinite(lnlike)] = -np.inf

            # NB emission prob. under best state (includes phase flip complement).
            best_lnlike = np.max(lnlike, axis=0)
            total_best_lnlike = best_lnlike.sum()

            # NB >>1 where current emission states are not a good fit.
            ps = -best_lnlike

            # NB known normal baseline, zero prob. where it is not defined.
            if known_normal:
                ps[base_nb_mean == 0.0] = 0.0

            if anneal:
                thres = np.percentile(ps, 100.0 * 1.0 - (len(log_mu) / (n_states - 1)))
                ps[ps < thres] = 0.0

            ps /= ps.sum()
            ps = ps.ravel()

            sample_ln_rdr, sample_baf = np.inf, np.inf

            while (
                ((~np.isfinite(sample_ln_rdr) and known_normal))
                or ~np.isfinite(sample_baf)
                or sample_ln_rdr > np.log(max_rdr)
            ):
                sample_idx = np.random.choice(flat_idx, p=ps)

                sample_segment, sample_spot = (
                    sample_idx // num_spots,
                    sample_idx % num_spots,
                )

                # TODO base_nb_mean is (N,1)?
                sample_ln_rdr = np.log(
                    X[sample_segment, 0, sample_spot] / base_nb_mean[sample_segment, 0]
                )
                sample_baf = (
                    X[sample_segment, 1, sample_spot] / total_bb_RD[sample_segment, 0]
                )

            log_mu = np.vstack([log_mu, [[sample_ln_rdr]]])
            p_binom = np.vstack([p_binom, [[sample_baf]]])

            if (len(log_mu) == n_states) and total_best_lnlike > solution_lnlike:
                solution = [log_mu, alphas, p_binom, taus]
                solution_lnlike = total_best_lnlike

    return solution, solution_lnlike


def cna_mixture_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    anneal=True,
    width=1,
    max_iter=500,
    only_minor=False,
    max_rdr=np.inf,
    num_jobs=3,
):
    # TODO X is a clone stack along axis 0.
    if width is not None:
        X = top_hat_sum(X, width)[::width]
        base_nb_mean = top_hat_sum(base_nb_mean, width)[::width]
        total_bb_RD = top_hat_sum(total_bb_RD, width)[::width]

    known_normal_frac = np.mean(base_nb_mean > 0.0)
    known_normal = known_normal_frac > 0.0

    # NB reduce grid search space
    if known_normal:
        grid_alphas = np.logspace(-2, -1, 3, base=10.0)
    else:
        grid_alphas = np.array([1.0e-2])

    grid_taus = np.logspace(2, 3, 3, base=10.0)

    param_grid = [(alpha, tau) for alpha in grid_alphas for tau in grid_taus]

    logger.info(
        f"Initializing HMM emission with CNA Mixture++ for max_iter={max_iter}, num_grid_points={len(param_grid)}, num_jobs={num_jobs}, known normal={known_normal}, max_rdr={max_rdr}, with X.shape={X.shape}, base_nb_mean.shape={base_nb_mean.shape}."
    )

    results = Parallel(n_jobs=num_jobs)(
        delayed(cna_mixture_init_search)(
            alpha,
            tau,
            max_iter,
            anneal,
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            max_rdr,
            known_normal,
        )
        for alpha, tau in param_grid
    )

    (log_mu, alphas, p_binom, taus), solution_lnlike = max(
        results, key=lambda row: row[-1]
    )

    if only_minor:
        p_binom = np.where(p_binom > 0.5, 1.0 - p_binom, p_binom)

    if not (known_normal):
        log_mu, alphas = None, None

    logger.info(
        f"Solved for initial parameters with  copy mixture++ and lnlike={solution_lnlike:.6e}:\nlog_mu={log_mu},\nalphas={alphas},\np_binom={p_binom},\ntaus={taus}."
    )

    return log_mu, alphas, p_binom, taus


# TODO define width
def plot_cna_mixture(
    init_log_mu,
    init_alphas,
    init_p_binom,
    init_taus,
    X,
    base_nb_mean,
    total_bb_RD,
    width=1,
    prefix="initial",
    max_rdr=None,
):
    # NB base_nb_mean is zero until post-BAF normal identication; in which case,
    #    these will be NAN.
    X_gmm_rdr = np.vstack(
        [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
    ).T

    assert X_gmm_rdr.shape == (len(X[:, 0, 0]), X.shape[2])

    valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

    if np.all(~valid):
        X_gmm_rdr[~valid] = np.random.normal(
            loc=1.0, scale=0.01, size=np.count_nonzero(~valid)
        )

    # TODO clipping?
    X_gmm_baf = np.vstack(
        [
            top_hat_sum(X[:, 1, s], width) / top_hat_sum(total_bb_RD[:, s], width)
            for s in range(X.shape[2])
        ]
    ).T

    if init_p_binom is None:
        init_p_binom, _ = get_betabinom_start_params()
        init_p_binom = np.tile(np.array(init_p_binom).reshape(-1, 1), (1, X.shape[2]))

    if init_log_mu is None:
        init_log_mu, _ = get_nbinom_start_params()
        init_log_mu = np.array(init_log_mu).reshape(-1, 1)
        init_log_mu = np.tile(init_log_mu, (1, X.shape[2]))

    if init_alphas is None:
        config = get_global_config()
        init_alphas = config.nbinom.start_disp * np.ones_like(init_log_mu)
        init_alphas = np.tile(init_alphas, (1, X.shape[2]))

    if init_taus is None:
        config = get_global_config()
        init_taus = config.betabinom.start_disp * np.ones_like(init_p_binom)
        init_taus = np.tile(init_taus, (1, X.shape[2]))

    init_mu = np.exp(init_log_mu)

    num_clones, num_segments = X.shape[2], X.shape[0]

    # NB S,n = 3,4 ... [0 1 2 0 1 2 0 1 2 0 1 2], i.e. column major.
    clone_idx = np.tile(np.arange(num_clones), num_segments)

    palette = sns.color_palette(n_colors=num_clones)

    x = X_gmm_baf.ravel()
    y = X_gmm_rdr.ravel()

    g = sns.JointGrid(x=x, y=y, height=8, ratio=3, space=0.15)
    valid_mask = np.isfinite(x) & np.isfinite(y)

    lnlike_rdr, lnlike_baf = hmm_sitewise.compute_emission_probability_nb_betabinom(
        X, base_nb_mean, init_log_mu, init_alphas, total_bb_RD, init_p_binom, init_taus
    )

    lnlike = lnlike_baf + lnlike_rdr

    # TODO track finite.
    lnlike[~np.isfinite(lnlike)] = -np.inf

    # NB emission prob. under (0.0, 0.5)
    # best_lnlike = lnlike[0,:,:].ravel()

    # NB emission prob. under best state (includes phase flip complement).
    best_lnlike = np.max(lnlike, axis=0).ravel()

    like_ratio = np.exp(best_lnlike - best_lnlike.max())

    # alpha = 0.2 + (like_ratio - like_ratio.min()) * 0.8 / (1.0 - like_ratio.min())
    alpha = like_ratio

    for c in range(num_clones):
        clone_mask = (clone_idx == c) & valid_mask

        if np.any(clone_mask):
            g.ax_joint.scatter(
                x[clone_mask],
                y[clone_mask],
                s=1,
                marker=".",
                alpha=alpha,
                color=palette[c],
            )

    g.ax_joint.scatter(
        init_p_binom, init_mu, marker="*", facecolor="none", edgecolor="k", s=25
    )

    if max_rdr is not None:
        g.ax_joint.set_ylim(-1, max_rdr)

    xticks = np.arange(0.0, 1.05, 0.05)
    g.ax_joint.set_xticks(xticks)
    g.ax_joint.set_xticklabels(
        [f"{x:.2f}" if ((1 + ii) % 2) else "" for ii, x in enumerate(xticks)]
    )

    bins = 50

    validx = np.isfinite(x)
    validy = np.isfinite(y)

    bins_x = np.arange(-0.01, 1.0, 5.0e-3)
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
                label=cast_clone_label(f"clone {c}") if num_clones > 1 else "",
            )
        )

    g.set_axis_labels("ZHF", "RDR")
    g.ax_joint.legend(handles=legend_patches, loc="upper left", framealpha=0.0)

    fig = g.fig

    config = get_global_config()

    # {config.hmrf.n_clones_rdr}
    output_dir = f"{config.paths.output_dir}/clone{config.hmrf.n_clones}_rectangle{config.hmrf.random_state}_w{config.hmrf.spatial_weight:.1f}/"
    fig_path = f"{output_dir}/plots/{prefix}_rdr_baf.pdf"

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

        min_binom_prob = float(get_global_config().hmm.gmm_min_binom_prob)
        max_binom_prob = float(get_global_config().hmm.gmm_max_binom_prob)

        clipped = (X_gmm_baf < min_binom_prob) | (X_gmm_baf > max_binom_prob)

        logger.warning(
            f"Clipping {100. * np.mean(clipped):.4f} [%] of BAF values to [{min_binom_prob}, {max_binom_prob}]."
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
        f"Patched {num_patched/X_gmm.shape[1]:.4f} values with NaNs in input RDR/BAF data."
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

    logger.debug(f"Solved for GMM initialized parameters:\n{gmm_log_mu}\n{gmm_p_binom}")

    return gmm_log_mu, gmm_p_binom
