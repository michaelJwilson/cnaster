import copy
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import logging
import matplotlib.gridspec as gridspec
import cnaster.log_linear
from matplotlib.lines import Line2D
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.integer_copy import get_ordered_acn
from cnaster.utils import cast_clone_label

logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "DejaVu Serif"


# TODO immutable?
def get_ordered_acn():
    return [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (2, 2),
        (3, 1),
        (4, 0),
        (3, 2),
        (4, 1),
        (5, 0),
        (3, 3),
        (4, 2),
        (5, 1),
        (6, 0),
    ]


def get_full_palette(palette="tab20b"):
    colors = [
        "darkblue",
        "lightblue",
        "lightgray",
        "dimgray",
        "lightgoldenrodyellow",
        "gold",
        "navajowhite",
        "orange",
        "darkorange",
        "salmon",
        "red",
        "darkred",
        "plum",
        "orchid",
        "purple",
        "indigo",
    ]

    ordered_acn = get_ordered_acn()
    ordered_acn_rev = [xx[::-1] for xx in ordered_acn]

    # TODO HACK
    colors = sns.color_palette("tab20b", len(ordered_acn)).as_hex()
    # np.random.shuffle(colors)

    palette = dict(zip(ordered_acn, colors))

    """
    # TODO
    palette = {}
    palette.update({(0, 0): "darkblue"})
    palette.update({(1, 0): "lightblue"})
    palette.update({(1, 1): "lightgray", (2, 0): "dimgray"})
    palette.update({(2, 1): "lightgoldenrodyellow", (3, 0): "gold"})
    palette.update({(2, 2): "navajowhite", (3, 1): "orange", (4, 0): "darkorange"})
    palette.update({(3, 2): "salmon", (4, 1): "red", (5, 0): "darkred"})
    palette.update(
        {(3, 3): "plum", (4, 2): "orchid", (5, 1): "purple", (6, 0): "indigo"}
    )

    assert palette == new_palette
    """
    return palette, ordered_acn


def get_intervals(pred_cnv):
    """
    Find contiguous intervals in the array pred_cnv where the 
    copy number state is the same.  Returns the list of intervals
    and their state.
    """
    intervals, labs = [], []
    s = 0

    while s < len(pred_cnv):
        t = np.where(pred_cnv[s:] != pred_cnv[s])[0]
        if len(t) == 0:
            intervals.append((s, len(pred_cnv)))
            labs.append(pred_cnv[s])
            s = len(pred_cnv)
        else:
            t = t[0]
            intervals.append((s, s + t))
            labs.append(pred_cnv[s])
            s = s + t
    return intervals, labs

def plot_clones_genomic_simple(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    clone_index,
    lengths,
    single_tumor_prop=None,
    sample_list=None,
    remove_xticks=True,
    rdr_ylim=6,
    chrtext_shift=-0.2,
    base_height=3.2,
    pointsize=5,
    linewidth=1,
):
    logger.info("Plotting simplified RDR & BAF scatter plots per clone.")
    
    # Create pseudobulk for each clone
    X, base_nb_mean, total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )
    
    n_obs = X.shape[0]
    spots_per_clone = [len(idx) for idx in clone_index]
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]
    
    # Check if base_nb_mean is defined and has valid data
    has_rdr = base_nb_mean is not None and np.max(base_nb_mean) > 0
    
    n_pairs = len(nonempty_clones)
    axes_per_clone = 2 if has_rdr else 1  # RDR + BAF if RDR available, else just BAF
    n_axes_total = axes_per_clone * n_pairs
    
    fig = plt.figure(figsize=(20, base_height * n_pairs), dpi=300, facecolor="white")
    
    # Build height_ratios with spacing between pairs
    height_ratios = []
    for i in range(n_pairs):
        height_ratios.extend([1] * axes_per_clone)
        if i < n_pairs - 1:
            height_ratios.append(0.25)
    
    n_rows = len(height_ratios)
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0)
    
    axes, row = [], 0
    for i in range(n_axes_total):
        axes.append(fig.add_subplot(gs[row, 0]))
        row += 1
        if (i % axes_per_clone == axes_per_clone - 1) and (i < n_axes_total - 1):
            row += 1
    
    if sample_list is not None:
        fig.suptitle(", ".join(sample_list), x=0.5, y=0.99, fontsize=16, ha="center")
    
    unique_chrs = np.arange(len(lengths))
    
    for s, c in enumerate(nonempty_clones):
        ax_idx = s * axes_per_clone
        
        if has_rdr:
            sns.scatterplot(
                x=np.arange(X.shape[0]),
                y=X[:, 0, c] / base_nb_mean[:, c],
                s=pointsize,
                edgecolor="none",
                linewidth=linewidth,
                ax=axes[ax_idx],
            )
            
            axes[ax_idx].set_ylabel("RDR")
            axes[ax_idx].set_ylim([-0.5, rdr_ylim])
            axes[ax_idx].set_xlim([0, n_obs])
            
            if remove_xticks:
                axes[ax_idx].set_xticks([])
            
            # Add gray axvlines for segments in RDR
            for i in range(len(lengths)):
                axes[ax_idx].axvline(x=np.sum(lengths[:(i)]), c="lightgray", linewidth=0.5)
        
        baf_idx = ax_idx + (1 if has_rdr else 0)
        sns.scatterplot(
            x=np.arange(X.shape[0]),
            y=X[:, 1, c] / total_bb_RD[:, c],
            s=pointsize,
            edgecolor="none",
            alpha=0.8,
            legend=False,
            ax=axes[baf_idx],
        )
        
        axes[baf_idx].set_ylabel("BAF")
        axes[baf_idx].set_ylim([-0.05, 1.05])
        axes[baf_idx].set_yticks(np.arange(0.0, 1.1, 0.2))
        axes[baf_idx].set_xlim([0, n_obs])
        
        if remove_xticks:
            axes[baf_idx].set_xticks([])
        
        # Add gray axvlines for segments in BAF
        for i in range(len(lengths)):
            axes[baf_idx].axvline(x=np.sum(lengths[:(i)]), c="lightgray", linewidth=0.5)
        
        ax = axes[ax_idx]
        ax.text(
            -0.04,
            0.00,
            f"{cast_clone_label(str(c))}",
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
        )
        
        ax.text(
            0.0,
            1.02,
            f"{spots_per_clone[c]:_} spots; {int(np.sum(X[:, 0, c])):_} UMIs; {int(np.sum(total_bb_RD[:, c])):_} SNP-UMIs",
            ha="left",
            va="bottom",
            fontsize=12,
            transform=ax.transAxes,
        )
    
    for i in range(len(lengths)):
        median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[: (i + 1)]) * 0.45
        axes[-1].text(
            median_len - 7.5,
            chrtext_shift,
            f"chr{unique_chrs[i]}",
            transform=axes[-1].get_xaxis_transform(),
            fontsize=9,
            ha="left",
        )
        for k in range(len(axes)):
            axes[k].axvline(x=np.sum(lengths[:(i)]), c="k", linewidth=1)
    
    fig.tight_layout()
    return fig

def plot_clones_genomic(
    df_cnv,  # NB integer copy numbers for each segment.
    lengths,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res_combine,
    single_tumor_prop=None,
    clone_ids=None,
    clone_index=None,
    sample_list=None,
    remove_xticks=True,
    rdr_ylim=6,
    chrtext_shift=-0.2,
    base_height=3.2,
    pointsize=5,
    linewidth=1,
    palette_name="chisel",
):
    logger.info(f"Plotting inferred rdr & baf for all clones.")

    chisel_palette, ordered_acn = get_full_palette(palette_name)

    map_cn = {x: i for i, x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])

    # NB add in normal clone.
    if "0" not in final_clone_ids:
        logger.error("Pre-pending 0 to final_clone_ids")
        final_clone_ids = np.array(["0"] + list(final_clone_ids))

    assert (clone_ids is None) or np.all(
        [(cid in final_clone_ids) for cid in clone_ids]
    )

    # TODO?
    assert clone_ids is None

    n_states = res_combine["new_p_binom"].shape[0]
    unique_chrs = np.unique(df_cnv.CHR.values)

    # NB number of genomic segments conserved.
    assert single_X.shape[0] == df_cnv.shape[0]

    if clone_index is None:
        clone_index = [
            np.where(res_combine["new_assignment"] == c)[0]
            for c, _ in enumerate(final_clone_ids)
        ]

    # NB create pseudobulk for each clone.
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

    n_axes = 2 * len(nonempty_clones)  # RDR + BAF for each clone
    n_pairs = len(nonempty_clones)
    fig = plt.figure(figsize=(20, base_height * n_pairs), dpi=300, facecolor="white")

    # Build height_ratios: [1, 1, 2, 1, 1, 2, ...] (no space within pair, double space between pairs)
    height_ratios = []

    for i in range(n_pairs):
        height_ratios.extend([1, 1])  # No space between the pair

        if i < n_pairs - 1:
            height_ratios.append(0.25)  # Double space between pairs

    n_rows = len(height_ratios)
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0)

    axes, row = [], 0

    for i in range(n_axes):
        axes.append(fig.add_subplot(gs[row, 0]))
        row += 1

        # After every pair, skip the extra space row
        if (i % 2 == 1) and (i < n_axes - 1):
            row += 1

    if sample_list is not None:
        fig.suptitle(", ".join(sample_list), x=0.5, y=0.99, fontsize=16, ha="center")

    logger.info(
        f"Found non-empty clones: {nonempty_clones} for final_clone_ids={final_clone_ids}"
    )

    for s, c in enumerate(nonempty_clones):
        cid = final_clone_ids[c]

        # NB major & minor allele copies give the hue
        major = np.maximum(
            df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
        )
        minor = np.minimum(
            df_cnv[f"clone{cid} A"].values, df_cnv[f"clone{cid} B"].values
        )

        segments, labs = get_intervals(res_combine["pred_cnv"][:, c])

        if palette_name == "chisel":
            hue = pd.Categorical(
                [map_cn[(major[i], minor[i])] for i in range(len(major))],
                categories=np.arange(len(ordered_acn)),
                ordered=True,
            )
            palette = sns.color_palette(colors)
            logger.info(
                f"Assuming chisel, for clone {c} found unique copy number states: {set([(major[i], minor[i]) for i in range(len(major))])} and unique categories {hue.unique()}"
            )
        else:
            hue = pd.Categorical(
                res_combine["pred_cnv"][:, c],
                categories=np.arange(n_states),
                ordered=True,
            )
            palette = palette
            logger.info(
                f"For clone {c} found unique copy number states: {np.unique(res_combine["pred_cnv"][:, c])} and unique categories {hue.unique()}"
            )
        """
        axes[2 * s].scatter(
            x=np.arange(X[:, 1, c].shape[0]),  # NB integer per segment.
            y=X[:, 0, c]
            / base_nb_mean[:, c],  # NB UMIs relative to normal baseline.
            hue=hue,
            palette = palette,
            s=pointsize,
            edgecolor="none",
            linewidth=linewidth,
        )
        """

        sns.scatterplot(
            x=np.arange(X[:, 1, c].shape[0]),  # NB integer per segment.
            y=X[:, 0, c] / base_nb_mean[:, c],  # NB UMIs relative to normal baseline.
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            linewidth=linewidth,
            ax=axes[2 * s],
        )

        # axes[2 * s].set_yscale("linlog", threshold=1.0, base=2.0)
        axes[2 * s].set_ylabel(f"\nRDR")

        # k = int(np.floor(np.log2(rdr_ylim)))
        # axes[2 * s].set_yticks(np.logspace(0, k, num=k + 1, base=2.0))

        axes[2 * s].set_ylim([-0.5, rdr_ylim])
        axes[2 * s].set_yticklabels([f"{y:.1f}" for y in axes[2 * s].get_yticks()])
        axes[2 * s].set_xlim([0, n_obs])

        if remove_xticks:
            axes[2 * s].set_xticks([])

        if palette_name == "chisel":
            hue = pd.Categorical(
                [map_cn[(major[i], minor[i])] for i in range(len(major))],
                categories=np.arange(len(ordered_acn)),
                ordered=True,
            )
            palette = sns.color_palette(colors)
        else:
            hue = pd.Categorical(
                res_combine["pred_cnv"][:, c],
                categories=np.arange(n_states),
                ordered=True,
            )
            palette = palette

        # NB plot phased b-allele frequency
        sns.scatterplot(
            x=np.arange(X[:, 1, c].shape[0]),  # NB integer per segment.
            y=X[:, 1, c] / total_bb_RD[:, c],  # NB BAF.
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            alpha=0.8,
            legend=False,
            ax=axes[2 * s + 1],
        )

        """
        sd = total_bb_RD[:, c]
        sd = np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)

        if sd.max() > 0:
            alpha = sd / np.median(total_bb_RD[np.isfinite(total_bb_RD)])
            alpha = np.clip(alpha, None, 1.0)
            alpha[alpha < 0.2] = 0.3
        else:
            alpha = np.zeros_like(sd)

        # map hue categories to base RGB colors, then inject per-point alpha
        codes = hue.codes
        base_colors = np.array(
            [mcolors.to_rgba(palette[i]) if i >= 0 else (0, 0, 0, 1.0) for i in codes],
            dtype=float,
        )
        base_colors[:, 3] = alpha
        """
        """
        axes[2 * s + 1].scatter(
            x=np.arange(X[:, 1, c].shape[0]),  # NB integer per segment.
            y=X[:, 1, c] / total_bb_RD[:, c],  # NB BAF.
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            linewidth=linewidth,
        )
        """
        sns.scatterplot(
            x=np.arange(X[:, 1, c].shape[0]),  # NB integer per segment.
            y=X[:, 1, c] / total_bb_RD[:, c],  # NB BAF.
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="none",
            alpha=0.8,
            legend=False,
            ax=axes[2 * s + 1],
        )

        axes[2 * s + 1].set_ylabel(f"\nBAF")
        axes[2 * s + 1].set_ylim([-0.05, 1.05])
        axes[2 * s + 1].set_yticks(np.arange(0.0, 1.1, 0.2))
        axes[2 * s + 1].set_xlim([0, n_obs])

        if remove_xticks:
            axes[2 * s + 1].set_xticks([])

        for i, seg in enumerate(segments):
            for to_plot in np.arange(-0.5, rdr_ylim, 0.5):
                axes[2 * s].plot(
                    seg,
                    [
                        to_plot,
                        to_plot,
                    ],
                    c="lightgray",
                    linewidth=0.5,
                )
            axes[2 * s].plot(
                seg,
                [
                    np.exp(res_combine["new_log_mu"][labs[i], c]),
                    np.exp(res_combine["new_log_mu"][labs[i], c]),
                ],
                c="k",
                linewidth=0.5,
            )
            axes[2 * s + 1].plot(
                seg,
                [
                    res_combine["new_p_binom"][labs[i], c],
                    res_combine["new_p_binom"][labs[i], c],
                ],
                c="k",
                linewidth=0.5,
            )

            # NB phase flip.
            axes[2 * s + 1].plot(
                seg,
                [
                    1.0 - res_combine["new_p_binom"][labs[i], c],
                    1.0 - res_combine["new_p_binom"][labs[i], c],
                ],
                c="k",
                linewidth=0.5,
                linestyle="--",
            )

            for to_plot in np.arange(0.0, 1.1, 0.1):
                axes[2 * s + 1].plot(
                    seg,
                    [
                        to_plot,
                        to_plot,
                    ],
                    c="lightgray",
                    linewidth=0.5,
                )

        # TODO filter based on clone aggregated hue.
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

        axes[2 * s].legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1, 1.25),
            ncol=len(legend_elements),
            frameon=False,
            bbox_transform=axes[2 * s].transAxes,
        )

    for i in range(len(lengths)):
        median_len = np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[: (i + 1)]) * 0.45
        axes[-1].text(
            median_len - 7.5,
            chrtext_shift,
            f"chr{unique_chrs[i]}",
            transform=axes[-1].get_xaxis_transform(),
            fontsize=9,
            ha="left",
        )
        for k in range(2 * len(nonempty_clones)):
            axes[k].axvline(x=np.sum(lengths[:(i)]), c="k", linewidth=1)

    for s, c in enumerate(nonempty_clones):
        ax = axes[2 * s]
        ax.text(
            -0.04,
            0.00,
            f"{cast_clone_label(final_clone_ids[c])}",
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
            transform=ax.transAxes,
        )

        np.arange(X[:, 1, c].shape[0])

        ax.text(
            0.0,
            1.02,
            f"{spots_per_clone[c]:_} spots;  {int(np.sum(X[:, 0, c])):_} umis; {int(np.sum(total_bb_RD[:, c])):_} snp-umis",
            ha="left",
            va="bottom",
            fontsize=12,
            transform=ax.transAxes,
        )

    fig.tight_layout()

    return fig


def plot_clones_spatial(
    coords,
    assignment,
    single_tumor_prop=None,
    sample_list=None,
    sample_ids=None,
    base_width=4,
    base_height=3,
    palette="rocket",  # "Set2"
):
    """
    Plot the spatial distribution of assigned clones for multiple slices/samples.
    """
    logger.info(f"Plotting inferred positions for all clones.")

    # NB shift coordinates across samples
    shifted_coords = copy.copy(coords)

    if sample_ids is not None:
        x_offset = 0

        for s, sname in enumerate(sample_list):
            index = np.where(sample_ids == s)[0]
            shifted_coords[index, 0] = shifted_coords[index, 0] + x_offset
            x_offset += np.max(coords[index, 0]) + 10

    # NB number of clones and samples
    final_clone_ids = np.unique(assignment[~assignment.isnull()].values)
    n_final_clones = len(final_clone_ids)
    n_samples = 1 if sample_list is None else len(sample_list)

    # NB remove nan of single_tumor_prop; assumes 0.5(!)
    if single_tumor_prop is not None:
        copy_single_tumor_prop = copy.copy(single_tumor_prop)
        copy_single_tumor_prop[np.isnan(copy_single_tumor_prop)] = 0.5

    fig, axes = plt.subplots(
        1, 1, figsize=(base_width * n_samples, base_height), dpi=300, facecolor="white"
    )

    if "clone 0" in final_clone_ids:
        colorlist = ["lightgrey"] + sns.color_palette(
            palette, n_final_clones - 1
        ).as_hex()
    else:
        colorlist = sns.color_palette(palette, n_final_clones).as_hex()

    for c, cid in enumerate(final_clone_ids):
        idx = np.where((assignment.values == cid))[0]

        if single_tumor_prop is None:
            sns.scatterplot(
                x=shifted_coords[idx, 0],
                y=-shifted_coords[idx, 1],
                s=10,
                color=colorlist[c],
                linewidth=0,
                legend=None,
                ax=axes,
            )
        else:
            this_full_cmap = sns.color_palette(
                f"blend:lightgrey,{colorlist[c]}", as_cmap=True
            )
            quantile_colors = this_full_cmap(
                np.array(
                    [
                        0,
                        np.min(copy_single_tumor_prop[idx]),
                        np.max(copy_single_tumor_prop[idx]),
                        1,
                    ]
                )
            )
            quantile_colors = [
                matplotlib.colors.rgb2hex(x) for x in quantile_colors[1:-1]
            ]
            this_cmap = sns.color_palette(
                f"blend:{quantile_colors[0]},{quantile_colors[-1]}", as_cmap=True
            )
            sns.scatterplot(
                x=shifted_coords[idx, 0],
                y=-shifted_coords[idx, 1],
                s=10,
                hue=copy_single_tumor_prop[idx],
                palette=this_cmap,
                linewidth=0,
                legend=None,
                ax=axes,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colorlist[c],
            label=cid,
            markersize=10,
        )
        for c, cid in enumerate(final_clone_ids)
    ]
    axes.legend(
        legend_elements,
        [cast_clone_label(cid) for cid in final_clone_ids],
        handlelength=0.1,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
    )
    # axes.axis("off")
    axes.set_title(",".join(sample_list), loc="left", fontsize=10)

    fig.tight_layout()

    return fig
