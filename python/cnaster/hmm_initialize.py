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
from cnaster.utils import top_hat_sum_pad, cast_clone_label
from cnaster.config import get_global_config
from cnaster.utils import write_fig
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
    t,
    X,
    base_nb_mean,
    total_bb_RD,
    hmm_class,
    only_minor=True,
):
    logger.info(f"Initializing HMM emission with CNA Mixture++.")

    eff_element = get_eff_element(t, n_states)
    eff_element = 1

    logger.info(
        f"Found effective genomic element={eff_element:.3f} for (t,K)=({t},{n_states}) and {X.shape[0]} total genomic elements"
    )

    # NB true of phasing partition and assumed below (currently), i.e. fed by one pseudo-bulk at a time.
    assert X.shape[-1] == 1

    # TODO HACK
    start_params = np.array([0.5, 1.0])
    settings = get_em_solver_params()

    # NB sample group from data.
    # TODO exclude existing states? rare clash?
    group_idx = int(np.floor(np.random.randint(low=0, high=X.shape[0]) // eff_element))
    start_idx = group_idx * eff_element
    end_idx = start_idx + eff_element

    num_groups = X.shape[0] // eff_element
    states = []

    while len(states) < n_states:
        # NB fit emission parameters to group
        endog = X[start_idx:end_idx, 1, :].flatten()
        exposure = total_bb_RD[start_idx:end_idx, ...].flatten()

        n_samples = len(endog)
        exog = np.ones((n_samples, 1))
        weights = np.ones(n_samples)

        solver = Weighted_BetaBinom_mix(endog, exog, weights, exposure)
        result = solver.fit(start_params=start_params, **settings)

        states.append(result.params)

        logger.info(f"Solved for {len(states)} states.")

        # NB evaluate data likelihood for current states
        endog = X[:, 1, :].flatten()
        exposure = total_bb_RD.flatten()

        n_samples = len(endog)
        exog = np.ones((n_samples, 1))
        weights = np.ones(n_samples)

        all_group_nlls = []

        for state in states:
            nll = nloglikeobs_bb(
                endog, exog, weights, exposure, state, tumor_prop=None, reduce=False
            )

            group_nlls = np.zeros(num_groups)

            for group_idx in range(num_groups):
                start_idx = group_idx * eff_element
                end_idx = start_idx + eff_element

                # NB assumes IDD for samples in each eff_element.
                group_nlls[group_idx] = np.sum(nll[start_idx:end_idx])

            all_group_nlls.append(group_nlls)

        all_group_nlls = np.column_stack(all_group_nlls).T
        best_group_nll = np.min(all_group_nlls, axis=0)

        best_group_idx = np.argmin(all_group_nlls, axis=0)

        print(states)
        print(np.unique(best_group_idx, return_counts=True))

        # NB sample next state.
        ps = best_group_nll / best_group_nll.sum()
        idx = np.random.choice(range(len(ps)), p=ps)

        start_idx = idx * eff_element
        end_idx = start_idx + eff_element

    p_binom = np.array([state[0] for state in states])

    # TODO
    tau = max([state[1] for state in states])

    if only_minor:
        p_binom = np.where(p_binom > 0.5, 1.0 - p_binom, p_binom)

    return None, p_binom


# TODO define width
def plot_cna_mixture(
    init_log_mu, init_p_binom, X, base_nb_mean, total_bb_RD, width=100, prefix="initial"
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
        X_gmm_rdr[~valid] = 1.0

    # TODO clipping?
    X_gmm_baf = np.vstack(
        [
            top_hat_sum_pad(X[:, 1, s], width)
            / top_hat_sum_pad(total_bb_RD[:, s], width)
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

    g.plot_joint(plt.scatter, s=1, marker=",", alpha=0.6, color="k")
    g.ax_joint.scatter(init_p_binom, init_mu, marker="*", c="gold", s=10)

    # Marginal histograms: use shared bin edges for comparability
    bins=50

    validx = np.isfinite(x)
    validy = np.isfinite(y)
    
    # bins_x = np.histogram_bin_edges(x[validx], bins=bins)
    # bins_y = np.histogram_bin_edges(y[validy], bins=bins)

    bins_x = np.arange(-0.05, 1.05, 0.01)
    bins_y = np.arange(-0.5, 10., 0.1)
    
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

    g.set_axis_labels("BAF", "RDR")
    g.ax_joint.legend(handles=legend_patches, loc="upper left", framealpha=0.0)

    fig = g.fig

    config = get_global_config()
    fig_path = f"{config.paths.output_dir}/plots/{prefix}_rdr_baf.pdf"

    logger.info(f"Writing initial copy state mixture plot to {fig_path}")

    write_fig(fig_path, fig, transparent=True, bbox_inches="tight")

    exit(0)


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
