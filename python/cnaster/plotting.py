import os
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.integer_copy import get_ordered_acn
from cnaster.utils import cast_clone_label, write_fig
from cnaster.config import get_global_config

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

def plot_gene_snp_spatial(
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    df_gene_snp,
    unique_snp_ids,
    plots_dir,
    pointsize=10,
    cmap="viridis",
    base_height=4,
    sampling=1.,
    max_genes=100,
):
    genes = df_gene_snp["gene"].unique()

    def get_gene_umi(g):
        if g in adata.var_names:
            return float(np.sum(adata[:, g].X))
        return -1.0

    genes = sorted(genes, key=get_gene_umi, reverse=True)

    logger.info(f"Plotting spatial distribution for {len(genes)} genes")

    coords = adata.obsm["X_pos"]

    os.makedirs(f"{plots_dir}/genes", exist_ok=True)

    gene_count = 0

    for gene_name in genes:
        if np.random.rand() > sampling:
            continue

        if gene_count >= max_genes:
            break

        gene_count += 1

        try:
            gene_expression = adata[:, gene_name].X
            if hasattr(gene_expression, "toarray"):
                gene_expression = gene_expression.toarray()
            gene_expression = np.array(gene_expression).flatten()
        except KeyError:
            logger.error(f"Gene {gene_name} not found in adata.")
            continue
        
        relevant_snps = df_gene_snp[df_gene_snp["gene"] == gene_name]["snp_id"].values
        
        if len(relevant_snps) == 0:
            logger.warning(f"No SNPs found for gene {gene_name} in df_gene_snp.")
            snp_A = np.zeros(coords.shape[0])
            snp_B = np.zeros(coords.shape[0])
        else:
            snp_indices = np.where(np.isin(unique_snp_ids, relevant_snps))[0]
            
            if len(snp_indices) == 0:
                 logger.warning(f"SNPs for {gene_name} found in table but not in matrix columns.")
                 snp_A = np.zeros(coords.shape[0])
                 snp_B = np.zeros(coords.shape[0])
            else:
                snp_A = np.array(cell_snp_Aallele[:, snp_indices].sum(axis=1)).flatten()
                snp_B = np.array(cell_snp_Ballele[:, snp_indices].sum(axis=1)).flatten()

        snp_total = snp_A + snp_B

        fig, axes = plt.subplots(1, 3, figsize=(base_height * 3.5, base_height), dpi=300, facecolor="white")
        
        total_umis = int(np.sum(gene_expression))
        total_snp_umis = int(np.sum(snp_total))

        titles = [
            f"{gene_name} umis: {total_umis:_}",
            f"{gene_name} snp-umis: {total_snp_umis:_}",
            f"{gene_name} $\\alpha$s: {int(np.sum(snp_A)):_}",
        ]
        data_layers = [gene_expression, snp_total, snp_A]

        for i, ax in enumerate(axes):
            data = data_layers[i]
            is_zero = data == 0

            if np.any(is_zero):
                ax.scatter(
                    coords[is_zero, 0],
                    -coords[is_zero, 1],
                    c="white",
                    s=pointsize,
                    edgecolor="black",
                    linewidth=0.1,
                    alpha=0.9,
                )

            if np.any(~is_zero):
                c = np.log10(data[~is_zero]) if i ==0 else data[~is_zero]
                sc = ax.scatter(
                    coords[~is_zero, 0],
                    -coords[~is_zero, 1],
                    c=c,
                    s=pointsize,
                    cmap=cmap,
                    edgecolor="none",
                    alpha=0.9,
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(sc, cax=cax)
                cbar.set_label("log10(count)" if i == 0 else "count")

            ax.set_title(titles[i], fontsize=12)
            ax.axis("off")
        
        fig.tight_layout()

        gene_fig_path = f"{plots_dir}/genes/{gene_name}_umis{total_umis}_snpumis{total_snp_umis}_spatial.pdf"
        write_fig(gene_fig_path, fig, transparent=True, bbox_inches="tight")

def plot_adjacency(
    coords,
    smooth_mat,
    adjacency_mat,
    pointsize=5,
    base_height=6,
    cmap="tab20b",
    sample_list=None
):
    fig, ax = plt.subplots(1, 1, figsize=(base_height * 1.2, base_height), dpi=300, facecolor="white")

    if sample_list is not None:
        ax.set_title(", ".join(sample_list) + ": adjacency", fontsize=12, y=0.95)
    else:
        ax.set_title("Adjacency", fontsize=12, y=0.95)

    ax.scatter(
        coords[:, 0],
        -coords[:, 1],
        facecolors="none",
        s=pointsize,
        edgecolor="k",
        linewidth=0.1,
        alpha=0.8,
        zorder=1
    )

    # NB can be pooled with self only.
    rows, cols = smooth_mat.nonzero()

    logger.info(f"Mean pooling per spot: {np.mean(smooth_mat.sum(axis=0))}")

    exclude = set()

    for i, j in zip(rows, cols):
        if i in exclude:
            continue

        exclude.add(j)

        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [-coords[i, 1], -coords[j, 1]],
            c="k",
            alpha=1.,
            linewidth=0.1,
            zorder=3
        )

    logger.info(f"Mean edge weight per spot: {np.mean(adjacency_mat.sum(axis=0))}")

    rows, cols = adjacency_mat.nonzero()
    weights = np.array(adjacency_mat[rows, cols]).flatten()
    
    max_weight = weights.max() if weights.size > 0 else 1.0
    
    cm = plt.get_cmap(cmap)
    n_nodes = coords.shape[0]

    node_colors = np.random.randint(0, 20, size=n_nodes)
    exclude = set()

    for _, (row, col, weight) in enumerate(zip(rows, cols, weights)):
        if row in exclude:
            continue

        exclude.add(col)

        # NB row & col guranteed to be in visited, with rank fixed by first appearance.
        c_idx = node_colors[row]
        ax.plot(
            [coords[row, 0], coords[col, 0]],
            [-coords[row, 1], -coords[col, 1]],
            c=cm(node_colors[c_idx]),
            alpha=1.0,
            linewidth=0.5 * weight / max_weight,
            zorder=1
        )

    ax.axis("off")
    
    fig.tight_layout()
    return fig

def plot_clones_genomic_simple(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    clone_index,
    lengths,
    res=None,
    single_tumor_prop=None,
    sample_list=None,
    remove_xticks=True,
    rdr_ylim=6,
    chrtext_shift=-0.1,
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
        fig.suptitle(", ".join(sample_list), x=0.5, y=0.98, fontsize=16, ha="center")
    
    unique_chrs = 1 + np.arange(len(lengths))
    
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

            for y in np.arange(0, rdr_ylim, 0.5):
                axes[ax_idx].axhline(y=y, c="lightgray", linewidth=0.5)

            if remove_xticks:
                axes[ax_idx].set_xticks([])
            
            for i in range(len(lengths)):
                axes[ax_idx].axvline(x=np.sum(lengths[:(i)]), c="black", linewidth=0.5)
        
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

        for y in np.arange(0.0, 1.1, 0.1):
            axes[baf_idx].axhline(y=y, c="lightgray", linewidth=0.5)
        
        if remove_xticks:
            axes[baf_idx].set_xticks([])
        
        for i in range(len(lengths)):
            axes[baf_idx].axvline(x=np.sum(lengths[:(i)]), c="black", linewidth=0.5)
        
        ax = axes[ax_idx]
        ax.text(
            -0.04,
            0.00 if has_rdr else 0.5,
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
            f"{spots_per_clone[c]:_} spots; {int(np.sum(X[:, 0, c])):_} umis; {int(np.sum(total_bb_RD[:, c])):_} snp-umis",
            ha="left",
            va="bottom",
            fontsize=12,
            transform=ax.transAxes,
        )

        if res is not None:
            # NB for all clones
            max_pred = np.argmax(res["log_gamma"], axis=0)
            this_pred = max_pred[(c * n_obs) : (c * n_obs + n_obs)]

            segments, labs = get_intervals(this_pred)
                
            mus = np.exp(res["new_log_mu"])
            ps = res["new_p_binom"]

            for i, (seg, state) in enumerate(zip(segments, labs)):
                if has_rdr:
                    ax.plot(
                        seg, 
                        [mus[state], mus[state]], 
                        c="k", 
                        linewidth=1.0
                    )
                
                axes[baf_idx].plot(
                    seg, 
                    [ps[state], ps[state]], 
                    c="k", 
                    linewidth=1.0
                )

                axes[baf_idx].plot(
                    seg, 
                    [1. - ps[state], 1. - ps[state]], 
                    c="k", 
                    linewidth=1.0,
                    linestyle="--",
                )
    
    for i in range(len(lengths)):
        start_len = np.sum(lengths[:(i)])
        axes[-1].text(
            start_len,
            chrtext_shift,
            f"chr{unique_chrs[i]}",
            rotation=45,
            transform=axes[-1].get_xaxis_transform(),
            fontsize=10,
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
    logger.info(f"Plotting inferred rdr+baf for all clones.")

    chisel_palette, ordered_acn = get_full_palette(palette_name)

    map_cn = {x: i for i, x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])

    # NB add in normal clone.
    if "0" not in final_clone_ids:
        logger.warning("Pre-pending 0 to final_clone_ids")
        final_clone_ids = np.array(["0"] + list(final_clone_ids))

    assert (clone_ids is None) or np.all(
        [(cid in final_clone_ids) for cid in clone_ids]
    )

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

    # TODO?
    assert clone_ids is None

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

def plot_recombination_rates(df_recomb, base_height=4):
    df = df_recomb.copy()
    df['chrom'] = df['chrom'].astype(str).str.replace('chr', '')
    
    valid_chroms = [str(i) for i in range(1, 23)]
    df = df[df['chrom'].isin(valid_chroms)]
    df['chrom'] = df['chrom'].astype(int)
    df = df.sort_values(['chrom', 'pos'])

    """
    chrom_mins = df.groupby('chrom')['pos'].min()
    chrom_maxes = df.groupby('chrom')['pos'].max()

    
    logger.info("Contig ranges:")
    for chrom in chrom_mins.index:
        logger.info(f"chr{chrom:<2}:\t{chrom_mins[chrom]:>12_} - {chrom_maxes[chrom]:>12_}")
    """
        
    unique_chroms = sorted(df['chrom'].unique())
    n_chroms = len(unique_chroms)

    fig, axes = plt.subplots(
        n_chroms, 
        1, 
        figsize=(15, max(base_height, n_chroms * 0.8)), 
        sharex=True, 
        sharey=True, 
        dpi=300
    )
    
    if n_chroms == 1:
        axes = [axes]

    for i, chrom in enumerate(unique_chroms):
        chrom_data = df[df['chrom'] == chrom].copy()
        chrom_data.loc[:, "pos"] = chrom_data["pos"] / 1e6  # Convert to Mb

        ax = axes[i]
        
        sns.lineplot(
            data=chrom_data,
            x='pos',
            y='recomb_rate',
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
            c="k", 
        )

        ax.set_ylabel(f"chr{chrom}", rotation=90, ha='right', va='bottom', fontsize=10)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 100)

        sns.despine(ax=ax)
        
        if i < n_chroms - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Pos [Mb]")
    
    fig.suptitle("Recombination rate")
    plt.tight_layout()
    
    return fig

def plot_copy_states(state_cnv):
    clone_cols = [c for c in state_cnv.columns if "logmu" in c]
    clone_names = sorted(set(c.split()[0] for c in clone_cols))
    if len(clone_names) == 0:
        logger.warning("No clone columns detected in per-state table.")
        return
    n_states = len(state_cnv)

    print(state_cnv)

    # Collect unique (A,B) states across ALL clones for global color palette
    global_states = set()
    for clone in clone_names:
        a_vals = state_cnv[f"{clone} A"].astype(int).to_list()
        b_vals = state_cnv[f"{clone} B"].astype(int).to_list()
        for a, b in zip(a_vals, b_vals):
            global_states.add((a, b))

    # Build global color palette
    ordered_global_states = sorted(
        global_states,
        key=lambda ab: (
            ab[0] + ab[1],
            ab[0] / (ab[0] + ab[1]) if (ab[0] + ab[1]) > 0 else 0,
        ),
    )
    palette = sns.color_palette("husl", len(ordered_global_states))
    state_colors = {st: palette[i] for i, st in enumerate(ordered_global_states)}
    state_colors[(1, 1)] = "#FFFFFF"

    # Order HMM states independently per clone by BAF then μ
    # Build a dict: clone -> sorted list of state indices
    clone_state_orders = {}

    def _state_cmp(a, b):
        # a, b: (state_index, baf, mu, (A,B))
        if abs(a[1] - b[1]) < 0.05:
            return -1 if a[2] < b[2] else (1 if a[2] > b[2] else 0)
        return -1 if a[1] < b[1] else (1 if a[1] > b[1] else 0)

    for clone in clone_names:
        order_info = []
        for s in range(n_states):
            baf = state_cnv.iloc[s][f"{clone} p"]
            logmu = state_cnv.iloc[s][f"{clone} logmu"]
            mu = np.exp(logmu)
            a = int(state_cnv.iloc[s][f"{clone} A"])
            b = int(state_cnv.iloc[s][f"{clone} B"])
            order_info.append((s, baf, mu, (a, b)))
        order_info = sorted(order_info, key=cmp_to_key(_state_cmp))
        clone_state_orders[clone] = [x[0] for x in order_info]

    col_labels = [f"$\\mathbb{{R}}_{{{i}}}$" for i in range(n_states)]

    # Build table rows: 3 per clone (μ, BAF, (A,B)), using each clone's own state ordering.
    table_rows = []
    row_types = []
    for clone in clone_names:
        sorted_indices = clone_state_orders[clone]
        for rtype in (0, 1, 2):
            row = []
            for s_idx in sorted_indices:
                if rtype == 0:  # μ
                    logmu = state_cnv.iloc[s_idx][f"{clone} logmu"]
                    mu = np.exp(logmu)
                    row.append(f"{mu:.3f}")
                elif rtype == 1:  # BAF
                    baf = state_cnv.iloc[s_idx][f"{clone} p"]
                    row.append(f"{baf:.3f}")
                else:  # (A,B)
                    a = int(state_cnv.iloc[s_idx][f"{clone} A"])
                    b = int(state_cnv.iloc[s_idx][f"{clone} B"])
                    row.append(f"({a},{b})")
            table_rows.append(row)
            row_types.append(rtype)

    fig_height = max(6, len(clone_names) * 3 * 0.35 + 2)
    fig_width = max(10, len(col_labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    fig.canvas.draw()

    # Style header
    for c in range(len(col_labels)):
        cell = tbl[(0, c)]
        cell.set_facecolor("#FFFFFF")
        cell.set_text_props(fontsize=9)

    def blend(color, alpha=0.5):
        r, g, b, _ = mcolors.to_rgba(color)
        r = 1 - alpha * (1 - r)
        g = 1 - alpha * (1 - g)
        b = 1 - alpha * (1 - b)
        return (r, g, b, 1.0)

    data_row_offset = 1

    # Color all sub-rows by (A,B) state from global palette (using each clone's ordering)
    for r, rtype in enumerate(row_types):
        table_r = r + data_row_offset
        clone_idx = (
            r // 3
        )  # NB assumes three row sub-types i.e. (μ, BAF, (A,B)) per clone.
        clone = clone_names[clone_idx]
        sorted_indices = clone_state_orders[clone]
        for c, s_idx in enumerate(sorted_indices):
            a = int(state_cnv.iloc[s_idx][f"{clone} A"])
            b = int(state_cnv.iloc[s_idx][f"{clone} B"])
            base_col = state_colors.get((a, b), "#FFFFFF")
            tbl[(table_r, c)].set_facecolor(blend(base_col, alpha=0.5))

    for clone_idx, clone in enumerate(clone_names):
        display_clone = cast_clone_label(clone)
        start_r = data_row_offset + clone_idx * 3
        top_cell = tbl[(start_r, 0)]
        bottom_cell = tbl[(start_r + 2, 0)]
        y_center = (
            top_cell.get_y() + bottom_cell.get_y() + bottom_cell.get_height()
        ) / 2
        ax.text(
            -0.050,
            y_center,
            display_clone,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            transform=ax.transAxes,
        )

    # Vertical sub-row labels
    label_map = {0: r"$\mu$", 1: r"$\beta$", 2: r"$\mathbb{N}$"}
    for r, rtype in enumerate(row_types):
        table_r = r + data_row_offset
        first_cell = tbl[(table_r, 0)]
        y_center = first_cell.get_y() + first_cell.get_height() / 2
        ax.text(
            -0.020,
            y_center,
            label_map[rtype],
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            transform=ax.transAxes,
        )

    plt.title(r"$\mathbb{R}$ copy states", fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig