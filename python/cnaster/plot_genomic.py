import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.utils import cast_clone_label, get_intervals
from cnaster.palette import get_full_palette

logger = get_logger(__name__, start_time=start_time)


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
    # A uniform negative horizontal offset ensures alignment is consistently
    # to the left of the RDR/BAF y-labels. Adjust -0.08 if you need more/less gap.
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
        # Align horizontally to the left of the RDR/BAF labels and vertically
        # to the axis line separating the top (RDR) and bottom (BAF) plots.
        # y=0.0 in the top ax's coordinates corresponds exactly to that separating line.
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


def plot_clones_genomic_raw(
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    clone_index: list,
    lengths: np.ndarray,
    res: dict = None,
    single_tumor_prop: np.ndarray = None,
    sample_list: list = None,
    remove_xticks: bool = True,
    rdr_ylim: float = 6.0,
    chrtext_shift: float = -0.2,
    base_height: float = 3.2,
    pointsize: float = 5.0,
    linewidth: float = 1.0,
):
    """
    Plots Read-Depth Ratio (RDR) and B-Allele Frequency (BAF) for multiple clones
    without enforcing explicit integer copy number states or color mappings.
    """
    logger.info("Plotting rdr & baf data (per clone) w/o fits.")

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    n_obs = X.shape[0]
    spots_per_clone = [len(idx) for idx in clone_index]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]
    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0
    unique_chrs = 1 + np.arange(len(lengths))

    n_pairs = len(nonempty_clones)
    axes_per_clone = 2 if has_rdr else 1
    fig, axes = _create_clone_gridspec(
        n_pairs, axes_per_clone, base_height, sample_list
    )

    for s, c in enumerate(nonempty_clones):
        ax_idx = s * axes_per_clone

        if has_rdr:
            ax_rdr = axes[ax_idx]
            sns.scatterplot(
                x=np.arange(n_obs),
                y=X[:, 0, s] / base_nb_mean[:, s],
                s=pointsize,
                edgecolor="none",
                linewidth=linewidth,
                ax=ax_rdr,
                zorder=1,
            )
            _format_track_axis(
                ax_rdr,
                "\nRDR",
                [-0.5, rdr_ylim],
                np.arange(0, rdr_ylim + 1.0, 1.0),
                remove_xticks,
                n_obs,
            )

        # Plot BAF
        baf_idx = ax_idx + (1 if has_rdr else 0)
        ax_baf = axes[baf_idx]
        sns.scatterplot(
            x=np.arange(n_obs),
            y=X[:, 1, s] / total_bb_RD[:, s],
            s=pointsize,
            edgecolor="none",
            alpha=0.8,
            legend=False,
            ax=ax_baf,
            zorder=1,
        )
        _format_track_axis(
            ax_baf,
            "\nBAF",
            [-0.05, 1.05],
            np.arange(0.0, 1.1, 0.2),
            remove_xticks,
            n_obs,
        )

        # Annotations
        t_prop = tumor_prop[s] if single_tumor_prop is not None else None
        _annotate_clone_stats(
            axes[ax_idx],
            c,
            spots_per_clone[s],
            np.sum(X[:, 0, s]),
            np.sum(total_bb_RD[:, s]),
            t_prop,
            paired_ax=ax_baf if has_rdr else None,
        )

        # HMM Model Fits
        if res is not None:
            max_pred = np.argmax(res["log_gamma"], axis=0)
            this_pred = max_pred[(s * n_obs) : (s * n_obs + n_obs)] % res["n_states"]

            # NB currently based on _inferred real state_, as opposed to integer (A,B) states,
            segments, labels = get_intervals(this_pred)

            mus = np.exp(res["new_log_mu"])
            ps = res["new_p_binom"]

            for i, (seg, state) in enumerate(zip(segments, labels)):
                if has_rdr:
                    ax_rdr.plot(
                        seg, [mus[state], mus[state]], c="k", linewidth=1.0, zorder=2
                    )

                ax_baf.plot(seg, [ps[state], ps[state]], c="k", linewidth=1.0, zorder=2)
                ax_baf.plot(
                    seg,
                    [1.0 - ps[state], 1.0 - ps[state]],
                    c="k",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=2,
                )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)
    fig.tight_layout()
    return fig


def plot_clones_genomic(
    df_cnv: pd.DataFrame, # segment level: chr, start, end, real states (Z), A/B copies, & model (log_mu, p_binom) for each clone.
    lengths: np.ndarray,
    single_X: np.ndarray,
    single_base_nb_mean: np.ndarray,
    single_total_bb_RD: np.ndarray,
    res_combine: dict,
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
    Plots fully modeled Read-Depth Ratio (RDR) and B-Allele Frequency (BAF) across clones,
    overlaying categorical copy number states and distribution errors.

    Error Models:
    - RDR: Assumes Poisson variance ($std = \sqrt{N} / N_{base}$).
    - BAF: Beta posterior uniform prior ($Beta(k+1, n-k+1)$) or Wald interval.
    """
    logger.info(f"Plotting inferred rdr+baf for all clones.")

    if plot_baf_errors not in (None, "wald", "beta"):
        raise ValueError("plot_baf_errors must be one of None, 'wald', or 'beta'")
    if plot_rdr_errors not in (None, "poisson"):
        raise ValueError("plot_rdr_errors must be one of None, or 'poisson'")

    # NB get palette map for copy number states, either (A,B)-like, or integer states.
    color_palette, ordered_acn = get_full_palette(palette_name)

    # NB mapper to enumeration for copy states.
    map_cn = {x: i for i, x in enumerate(ordered_acn)}

    # NB list of colors for each copy state.
    colors = [color_palette[c] for c in ordered_acn]

    # TODO BUG more robust extraction; expect "clone{cid} A" etc.,
    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])

    # 
    # if "0" not in final_clone_ids:
    #     final_clone_ids = np.array(["0"] + list(final_clone_ids))

    assert "0" in final_clone_ids

    # NB ambiguous on clone stack.
    n_states = res_combine["new_p_binom"].shape[0]
    unique_chrs = np.unique(df_cnv.CHR.values)

    assert single_X.shape[0] == df_cnv.shape[0], "Found mismatch for genomic segment defined X and derived copy state profiles."

    # DEPRECATE clone_index: expects spot indices for each clone;
    if clone_index is None:
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0]
            for c, _ in enumerate(final_clone_ids)
        ]

    X, base_nb_mean, total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    n_obs = X.shape[0]
    spots_per_clone = [len(xx) for xx in clone_index]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    assert len(nonempty_clones) == total_bb_RD.shape[1]

    fig, axes = _create_clone_gridspec(
        len(nonempty_clones), 2, base_height, sample_list
    )

    for s, c in enumerate(nonempty_clones):
        # NB derived from provided df_cnv
        cid = final_clone_ids[c]
        ax_rdr, ax_baf = axes[2 * s], axes[2 * s + 1]

        # NB best _REAL_ (not integer) copy states for this clone; run length encoded.
        #    will be assigned same color according to inferred integer (A,B) copies.
        segments, labels = get_intervals(res_combine["pred_cnv"][:, c])

        # NB major and minor copy numbers per segment for this clone.
        major = np.maximum(
            df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
        )
        minor = np.minimum(
            df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
        )

        if palette_name == "chisel":
            # NB colors are determined by major and minor only, not their order in (A,B) or (B,A).
            #    which would be chevroned.
            #
            # TODO simpler/more efficient way?
            hue = pd.Categorical(
                [map_cn[(major[i], minor[i])] for i in range(len(major))], # TODO runtime error if (major[i], minor[i]) not in map_cn; in lieu of ordered_acn.
                categories=np.arange(len(ordered_acn)), # NB color according to ordered_acn copy states.
                ordered=True,
            )

            # TODO more direct way?
            palette = sns.color_palette(colors)
        else:
            # NB no assumed color mapping; use __real__ copy states as categorical hue according to provided palette_name.
            hue = pd.Categorical(
                res_combine["pred_cnv"][:, c],
                categories=np.arange(n_states),
                ordered=True,
            )
            palette = palette_name

        # NB one per segment.
        x_vals = np.arange(n_obs)

        # TODO WTF??
        point_colors = [
            {i: palette[i] for i in range(len(palette))}[h] for h in hue.codes
        ]

        #  ----  RDR  ----

        # NB normal baseline scaled to total clone transcript count;
        y_vals_rdr = X[:, 0, c] / base_nb_mean[:, c]

        if plot_rdr_errors == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                # NB Poisson variance \propto mean; scaled by normal baseline.
                std_err_rdr = np.sqrt(X[:, 0, c]) / base_nb_mean[:, c]
                std_err_rdr[~np.isfinite(std_err_rdr)] = 0.0

            ax_rdr.errorbar(
                x_vals,
                y_vals_rdr,
                yerr=std_err_rdr,
                fmt="none",
                ecolor=point_colors,
                elinewidth=0.5,
                alpha=1.0,
                zorder=0,
            )

        sns.scatterplot(
            x=x_vals,
            y=y_vals_rdr,
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            linewidth=linewidth,
            ax=ax_rdr,
            zorder=1,
        )

        # NB generic / shared axis formatting for rdr and baf.
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

            # NB assumed uniform prior Beta(1,1); posterior is Beta(k+1, n-k+1).
            alpha, beta = k + 1, n - k + 1
            alpha_beta_sum = alpha + beta

            # TODO CHECK
            std_err_baf = np.sqrt(
                (alpha * beta) / (np.square(alpha_beta_sum) * (alpha_beta_sum + 1))
            )

            ax_baf.errorbar(
                x_vals,
                baf_vals,
                yerr=std_err_baf,
                fmt="none",
                ecolor=point_colors,
                elinewidth=0.5,
                alpha=1.0,
                zorder=0,
            )

        sns.scatterplot(
            x=x_vals,
            y=baf_vals,
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            alpha=0.8,
            legend=False,
            ax=ax_baf,
            zorder=1,
        )

        _format_track_axis(
            ax_baf,
            "\nBAF",
            [-0.05, 1.05],
            np.arange(0.0, 1.1, 0.2),
            remove_xticks,
            n_obs,
        )

        # NB plot model prediction for (run-length encoded) state, with lookup of best-fit
        #    params accroding to __real__ cna state.
        for i, seg in enumerate(segments):
            ax_rdr.plot(
                seg,
                [np.exp(res_combine["new_log_mu"][labels[i], c])] * 2,
                c="k",
                linewidth=0.5,
                zorder=2,
            )
            ax_baf.plot(
                seg,
                [res_combine["new_p_binom"][labels[i], c]] * 2,
                c="k",
                linewidth=0.5,
                zorder=2,
            )

            # NB phase switch.
            ax_baf.plot(
                seg,
                [1.0 - res_combine["new_p_binom"][labels[i], c]] * 2,
                c="k",
                linewidth=0.5,
                linestyle="--",
                zorder=2,
            )

        # NB state length with usage [%]. 
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
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

        # NB add useful clone statistics. 
        _annotate_clone_stats(
            ax_rdr,
            cid,
            spots_per_clone[c],
            np.sum(X[:, 0, c]),
            np.sum(total_bb_RD[:, c]),
            paired_ax=ax_baf,
        )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)
    
    fig.tight_layout()

    return fig
