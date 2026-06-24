import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.utils import cast_clone_label, get_intervals
from cnaster.palette import get_full_palette

logger = get_logger(__name__, start_time=start_time)

NORMAL_OPACITY = 0.75


def _create_clone_gridspec(
    n_pairs: int, axes_per_clone: int, base_height: float, sample_list: list = None
):
    """
    Creates a flexible GridSpec layout for plotting clones, automatically injecting
    vertical gaps between distinct clone tracks.
    """
    n_axes_total = axes_per_clone * n_pairs
    fig = plt.figure(figsize=(20, base_height * n_pairs), dpi=300, facecolor="white")

    height_ratios = []
    for i in range(n_pairs):
        height_ratios.extend([1] * axes_per_clone)
        if i < n_pairs - 1:
            height_ratios.append(0.25)  # Gap between clone tracks

    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    axes, row = [], 0
    for i in range(n_axes_total):
        axes.append(fig.add_subplot(gs[row, 0]))
        row += 1
        if (i % axes_per_clone == axes_per_clone - 1) and (i < n_axes_total - 1):
            row += 1  # Skip the gap row

    if sample_list is not None:
        fig.suptitle(", ".join(sample_list), x=0.5, y=0.99, fontsize=16, ha="center")

    return fig, axes


def _format_track_axis(ax, ylabel, ylim, yticks, remove_xticks, n_obs):
    """Standardizes the styling for rdr and baf genomic tracks."""
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])
    ax.set_xlim([0, n_obs])

    if remove_xticks:
        ax.set_xticks([])

    for y in yticks:
        ax.axhline(y=y, c="lightgray", linewidth=0.5, zorder=0)


def _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift):
    """Draws vertical contig boundaries and appends chromosome labels to the bottom axis."""
    for i in range(len(lengths)):
        start_len = np.sum(lengths[:i])

        # Label only on the bottom-most axis
        axes[-1].text(
            start_len,
            chrtext_shift,
            f"chr{unique_chrs[i]}",
            rotation=45,
            transform=axes[-1].get_xaxis_transform(),
            fontsize=10,
            ha="left",
        )
        # Draw boundaries across all axes
        for ax in axes:
            ax.axvline(x=start_len, c="black", linewidth=0.5)


def _annotate_clone_stats(
    ax,
    clone_label,
    n_spots,
    n_umis,
    n_snp_umis,
    tumor_prop=None,
    paired_ax=None,
):
    """Annotates the axis with clone identity and spot/UMI statistics."""
    x_offset = -0.04

    if paired_ax is None:
        ax.text(
            x_offset,
            0.5,
            cast_clone_label(str(clone_label)),
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
            clip_on=False,
        )
    else:
        ax.text(
            x_offset,
            0.0,
            cast_clone_label(str(clone_label)),
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
            clip_on=False,
        )

    theta_text = (
        f"$\\hat{{\\theta}}={tumor_prop:.2f}$" if tumor_prop is not None else ""
    )
    stats_text = f"{n_spots:_} spots; {int(n_umis):_} umis; {int(n_snp_umis):_} snp-umis; {theta_text}"

    ax.text(
        0.0,
        1.02,
        stats_text.strip("; "),
        ha="left",
        va="bottom",
        fontsize=12,
        transform=ax.transAxes,
    )


def plot_clones_genomic(
    df_cnv,  # Can be None for raw data plotting
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: dict = None,
    single_tumor_prop: np.ndarray = None,
    clone_ids: list = None,
    clone_index: list = None,
    sample_list: list = None,
    remove_xticks: bool = True,
    rdr_ylim: float = 6.0,
    chrtext_shift: float = -0.2,
    base_height: float = 3.2,
    pointsize: float = 3.0,
    linewidth: float = 1.0,
    palette_name: str = "chisel",
    plot_baf_errors: str = "beta",
    plot_rdr_errors: str = "poisson",
):
    """
    Plots aggregated rdr and baf (with error models) and best-fit continous copy states (mu, p).
    If df_cnv is None, functions as a raw data plotter without categorical integer states.
    """
    logger.info(f"Plotting aggregated rdr and baf for clones.")

    # Only extract palettes and mappings if we have categorical data
    if df_cnv is not None:
        color_palette, ordered_acn = get_full_palette(palette_name)
        map_cn = {x: i for i, x in enumerate(ordered_acn)}
        state_colors = [color_palette[c] for c in ordered_acn]

        final_clone_ids = (
            df_cnv.columns.str.extract(r"^clone(.*) A$", expand=False).dropna().tolist()
        )
        assert "0" in final_clone_ids
        unique_chrs = np.unique(df_cnv.CHR.values)
    else:
        unique_chrs = 1 + np.arange(len(lengths))
        if clone_ids is not None:
            final_clone_ids = clone_ids
        else:
            final_clone_ids = [str(i) for i in range(len(clone_index))]

    assert single_X.shape[0] == np.sum(
        lengths
    ), "Found mismatch for genomic segment defined X and lengths."

    if clone_index is None:
        assert (
            res_combine is not None
        ), "Must provide clone_index if res_combine is None"
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0]
            for c, _ in enumerate(final_clone_ids)
        ]

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    n_obs = X.shape[0]
    spots_per_clone = [len(xx) for xx in clone_index]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    # Determine if RDR data exists
    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0

    assert len(nonempty_clones) == total_bb_RD.shape[1]

    axes_per_clone = 2 if has_rdr else 1
    fig, axes = _create_clone_gridspec(
        len(nonempty_clones), axes_per_clone, base_height, sample_list
    )

    for s, c in enumerate(nonempty_clones):
        cid = final_clone_ids[c]

        ax_idx = s * axes_per_clone
        if has_rdr:
            ax_rdr = axes[ax_idx]
            ax_baf = axes[ax_idx + 1]
        else:
            ax_rdr = None
            ax_baf = axes[ax_idx]

        # Establish point colors and categorical hues
        if df_cnv is not None:
            major = np.maximum(
                df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
            )
            minor = np.minimum(
                df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
            )

            if palette_name == "chisel":
                default_idx = map_cn.get((1, 1), 0)
                hue_indices = [
                    map_cn.get((major[i], minor[i]), default_idx)
                    for i in range(len(major))
                ]

                hue = pd.Categorical(
                    hue_indices, categories=np.arange(len(ordered_acn)), ordered=True
                )
                palette = [
                    mcolors.to_rgba(
                        color,
                        alpha=(NORMAL_OPACITY if ordered_acn[i] == (1, 1) else 1.0),
                    )
                    for i, color in enumerate(state_colors)
                ]
            else:
                n_states = res_combine["new_p_binom"].shape[0]
                hue = pd.Categorical(
                    res_combine["pred_cnv"][:, c],
                    categories=np.arange(n_states),
                    ordered=True,
                )
                base_pal = sns.color_palette(palette_name, n_states)
                palette = [mcolors.to_rgba(color, alpha=1.0) for color in base_pal]

            point_colors = [palette[h] for h in hue.codes]
            scatter_kwargs = {"hue": hue, "palette": palette}
        else:
            point_colors = "#4C72B0"  # Default clean blue for raw plotting
            scatter_kwargs = {"color": point_colors}

        x_vals = np.arange(n_obs)

        #  ----  RDR  ----
        if has_rdr:
            y_vals_rdr = X[:, 0, c] / base_nb_mean[:, c]

            if plot_rdr_errors == "poisson":
                with np.errstate(divide="ignore", invalid="ignore"):
                    std_err_rdr = np.sqrt(X[:, 0, c]) / base_nb_mean[:, c]
                    std_err_rdr[~np.isfinite(std_err_rdr)] = 0.0

                ax_rdr.errorbar(
                    x_vals,
                    y_vals_rdr,
                    yerr=std_err_rdr,
                    fmt="none",
                    ecolor=point_colors if df_cnv is not None else "tab:blue",
                    elinewidth=0.5,
                    zorder=0,
                )

            sns.scatterplot(
                x=x_vals,
                y=y_vals_rdr,
                s=pointsize,
                edgecolor="none",
                linewidth=linewidth,
                legend=False,
                ax=ax_rdr,
                zorder=1,
                **scatter_kwargs,
            )

            _format_track_axis(
                ax_rdr,
                "\nRDR",
                [-0.5, rdr_ylim],
                np.arange(0, rdr_ylim + 1.0, 1.0),
                remove_xticks,
                n_obs,
            )

        # ----  BAF  ----
        baf_vals = X[:, 1, c] / total_bb_RD[:, c]

        if plot_baf_errors == "beta":
            k, n = X[:, 1, c], total_bb_RD[:, c]

            alpha_param, beta_param = k + 1, n - k + 1
            alpha_beta_sum = alpha_param + beta_param

            std_err_baf = np.sqrt(
                (alpha_param * beta_param)
                / (np.square(alpha_beta_sum) * (alpha_beta_sum + 1))
            )

            ax_baf.errorbar(
                x_vals,
                baf_vals,
                yerr=std_err_baf,
                fmt="none",
                ecolor=point_colors if df_cnv is not None else "tab:blue",
                elinewidth=0.5,
                zorder=0,
            )

        sns.scatterplot(
            x=x_vals,
            y=baf_vals,
            s=pointsize,
            edgecolor="none",
            legend=False,
            ax=ax_baf,
            zorder=1,
            **scatter_kwargs,
        )

        _format_track_axis(
            ax_baf,
            "\nBAF",
            [-0.05, 1.05],
            np.arange(0.0, 1.1, 0.2),
            remove_xticks,
            n_obs,
        )

        # ---- Model Prediction Lines ----
        if res_combine is not None:
            if df_cnv is not None:
                segments, labels = get_intervals(res_combine["pred_cnv"][:, c])
            else:
                # TODO HACK?
                max_pred = np.argmax(res_combine["log_gamma"], axis=0)
                this_pred = (
                    max_pred[(s * n_obs) : (s * n_obs + n_obs)] % res_combine["n_states"]
                )

                # NB currently based on _inferred real state_, as opposed to integer (A,B) states,
                segments, labels = get_intervals(this_pred)

            mus = np.exp(res_combine["new_log_mu"])
            ps = res_combine["new_p_binom"]
    
            # NB broadcast single fit across clones (to shape ... x len(nonempty_clones))                                                                                                                        
            if mus.shape[1] == 1:
                mus = np.repeat(mus, len(nonempty_clones), axis=1)
                ps = np.repeat(ps, len(nonempty_clones), axis=1)
                
            logger.info(f"Assuming model fits with mus.shape={mus.shape}.")

            for i, seg in enumerate(segments):
                if has_rdr:
                    ax_rdr.plot(
                        seg,
                        [mus[labels[i], s]] * 2,
                        c="k",
                        linewidth=0.5,
                        zorder=2,
                    )
                ax_baf.plot(
                    seg,
                    [ps[labels[i], s]] * 2,
                    c="k",
                    linewidth=0.5,
                    zorder=2,
                )
                ax_baf.plot(
                    seg,
                    [1.0 - ps[labels[i], s]] * 2,
                    c="k",
                    linewidth=0.5,
                    linestyle="--",
                    zorder=2,
                )

        # ---- Legend ----
        if df_cnv is not None and has_rdr:
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=state_colors[i],
                    label=f"{100. * np.mean(hue == i):.1f}% {ordered_acn[i]}",
                    markersize=10,
                    linestyle="None",
                )
                for i in hue.unique()
            ]

            ax_rdr.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1, 1.25),
                ncol=len(legend_elements),
                frameon=False,
                bbox_transform=ax_rdr.transAxes,
            )

        # ---- Clone Statistics Annotation ----
        t_prop = (
            tumor_prop[c]
            if (single_tumor_prop is not None and tumor_prop is not None)
            else None
        )

        _annotate_clone_stats(
            ax_rdr if has_rdr else ax_baf,
            cid,
            spots_per_clone[c],
            np.sum(X[:, 0, c]),
            np.sum(total_bb_RD[:, c]),
            t_prop,
            paired_ax=ax_baf if has_rdr else None,
        )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)

    fig.tight_layout()

    return fig
