import copy
import math
from functools import cmp_to_key

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.utils import cast_clone_label, write_fig

logger = get_logger(__name__, start_time=start_time)

plt.rcParams["font.family"] = "DejaVu Serif"


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
    sampling=1.0,
    max_genes=10,
):
    logger.info(
        f"Plotting spatial distribution for max_genes={max_genes} to {plots_dir}/genes"
    )

    # genes = df_gene_snp["gene"].unique()
    # genes = [g for g in genes if g in adata.var_names]

    genes = list(adata.var_names)

    def get_gene_umi(g):
        if g in adata.var_names:
            return float(np.sum(adata[:, g].X))
        return -1.0

    genes = sorted(genes, key=get_gene_umi, reverse=True)
    coords = adata.obsm["X_pos"]

    logger.info(f"Sorted input gene list.")

    # os.makedirs(f"{plots_dir}/genes", exist_ok=True)

    gene_count = 0

    for gene_name in genes:
        if np.random.rand() > sampling:
            continue

        if gene_count >= max_genes:
            break

        logger.info(f"Plotting gene={gene_name}")

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
                logger.warning(
                    f"SNPs for {gene_name} found in table but not in matrix columns."
                )
                snp_A = np.zeros(coords.shape[0])
                snp_B = np.zeros(coords.shape[0])
            else:
                snp_A = np.array(cell_snp_Aallele[:, snp_indices].sum(axis=1)).flatten()
                snp_B = np.array(cell_snp_Ballele[:, snp_indices].sum(axis=1)).flatten()

        snp_total = snp_A + snp_B

        fig, axes = plt.subplots(
            1, 3, figsize=(base_height * 3.5, base_height), dpi=300, facecolor="white"
        )

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
                c = np.log10(data[~is_zero]) if i == 0 else data[~is_zero]
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
    sample_list=None,
):
    fig, ax = plt.subplots(
        1, 1, figsize=(base_height * 1.2, base_height), dpi=300, facecolor="white"
    )

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
        zorder=1,
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
            alpha=1.0,
            linewidth=0.1,
            zorder=3,
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
            zorder=1,
        )

    ax.axis("off")

    fig.tight_layout()
    return fig


def plot_clones_spatial(
    coords,
    assignment,
    single_tumor_prop=None,
    sample_list=None,
    sample_ids=None,
    base_width=4,
    base_height=4,
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

        for s, _ in enumerate(sample_list):
            index = np.where(sample_ids == s)[0]
            shifted_coords[index, 0] = shifted_coords[index, 0] + x_offset
            x_offset += np.max(coords[index, 0]) + 10

    # NB number of clones and samples
    final_clone_ids = np.unique(assignment[~assignment.isnull()].values)
    n_final_clones = len(final_clone_ids)
    # n_samples = 1 if sample_list is None else len(sample_list)

    # NB remove nan of single_tumor_prop; assumes 0.5(!)
    if single_tumor_prop is not None:
        copy_single_tumor_prop = np.array(single_tumor_prop, dtype=float)
        invalid = np.isnan(copy_single_tumor_prop)

        if np.any(invalid):
            logger.warning(
                f"Imputing {100. * np.mean(invalid):.3f} [%] of NaN tumor proportion with 0.5"
            )
            copy_single_tumor_prop[np.isnan(copy_single_tumor_prop)] = 0.5

    # NB heuristic for marker size: 120000.0 is roughly appropriate for s=0.1 with ~100k spots.
    #    If we have fewer spots, we want larger markers.
    #    Clip to a reasonable range [0.1, 20].
    n_points = coords.shape[0]
    marker_size = np.clip(12_000.0 / n_points, 0.1, 25.0)

    fig, ax = plt.subplots(
        1, 1, figsize=(base_width, base_height), dpi=300, facecolor="white"
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
            ax.scatter(
                x=shifted_coords[idx, 0],
                y=-shifted_coords[idx, 1],
                s=marker_size,
                color=colorlist[c],
                linewidth=0,
            )
        else:
            vals = np.clip(copy_single_tumor_prop[idx], 0.0, 1.0)

            base_rgb = mcolors.to_rgb(colorlist[c])
            rgba_colors = np.zeros((len(vals), 4))
            rgba_colors[:, :3] = base_rgb
            rgba_colors[:, 3] = vals

            ax.scatter(
                shifted_coords[idx, 0],
                -shifted_coords[idx, 1],
                s=marker_size,
                c=rgba_colors,
                linewidth=0,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colorlist[c],
            label=cid,
            markersize=6,
        )
        for c, cid in enumerate(final_clone_ids)
    ]

    ax.legend(
        legend_elements,
        [cast_clone_label(cid) for cid in final_clone_ids],
        handlelength=0.1,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.02),  
        ncol=n_final_clones,
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,            
    )

    if sample_list is not None:
        ax.text(
            0.05,                      
            0.99,                     
            ", ".join(sample_list), 
            transform=ax.transAxes,
            ha="left", 
            va="bottom", 
            fontsize=9,
        )

    # ax.set_aspect("equal")
    ax.axis("off")

    # fig.tight_layout()

    return fig


def plot_recombination_rates(df_recomb, base_height=4):
    df = df_recomb.copy()
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "")

    valid_chroms = [str(i) for i in range(1, 23)]
    df = df[df["chrom"].isin(valid_chroms)]
    df["chrom"] = df["chrom"].astype(int)
    df = df.sort_values(["chrom", "pos"])

    """
    chrom_mins = df.groupby('chrom')['pos'].min()
    chrom_maxes = df.groupby('chrom')['pos'].max()

    
    logger.info("Contig ranges:")
    for chrom in chrom_mins.index:
        logger.info(f"chr{chrom:<2}:\t{chrom_mins[chrom]:>12_} - {chrom_maxes[chrom]:>12_}")
    """

    unique_chroms = sorted(df["chrom"].unique())
    n_chroms = len(unique_chroms)

    fig, axes = plt.subplots(
        n_chroms,
        1,
        figsize=(15, max(base_height, n_chroms * 0.8)),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    if n_chroms == 1:
        axes = [axes]

    for i, chrom in enumerate(unique_chroms):
        chrom_data = df[df["chrom"] == chrom].copy()
        chrom_data["pos"] = chrom_data["pos"].astype(float) / 1.0e6  # Convert to Mb

        ax = axes[i]

        sns.lineplot(
            data=chrom_data,
            x="pos",
            y="recomb_rate",
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
            c="k",
        )

        ax.set_ylabel(f"chr{chrom}", rotation=90, ha="right", va="bottom", fontsize=10)
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
    for i, clone in enumerate(clone_names):
        if i > 0:
            table_rows.append([""] * len(col_labels))
            row_types.append(-1)

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

    fig_height = max(6, len(table_rows) * 0.35 + 2)
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
    current_clone_idx = 0

    # Color all sub-rows by (A,B) state from global palette (using each clone's ordering)
    for r, rtype in enumerate(row_types):
        table_r = r + data_row_offset

        if rtype == -1:
            # Style the gap row: make it invisible
            for c in range(len(col_labels)):
                cell = tbl[(table_r, c)]
                cell.set_text_props(text="")
                cell.set_facecolor("none")
                cell.set_edgecolor("none")
                cell.set_height(0.02)  # Make gap row shorter
            continue

        clone = clone_names[current_clone_idx]
        sorted_indices = clone_state_orders[clone]

        for c, s_idx in enumerate(sorted_indices):
            a = int(state_cnv.iloc[s_idx][f"{clone} A"])
            b = int(state_cnv.iloc[s_idx][f"{clone} B"])
            base_col = state_colors.get((a, b), "#FFFFFF")
            tbl[(table_r, c)].set_facecolor(blend(base_col, alpha=0.5))

        if rtype == 2:
            current_clone_idx += 1

    current_row_idx = 0
    for clone_idx, clone in enumerate(clone_names):
        if clone_idx > 0:
            current_row_idx += 1  # Skip gap row

        display_clone = cast_clone_label(clone)
        start_r = data_row_offset + current_row_idx

        # Clone Label (Vertical Center of 3 rows)
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

        # Sub-row labels
        label_map = {0: r"$\mu$", 1: r"$\beta$", 2: r"$\mathbb{N}$"}
        for offset in range(3):
            table_r = start_r + offset
            first_cell = tbl[(table_r, 0)]
            y_center = first_cell.get_y() + first_cell.get_height() / 2
            ax.text(
                -0.020,
                y_center,
                label_map[offset],
                rotation=90,
                va="center",
                ha="center",
                fontsize=9,
                transform=ax.transAxes,
            )

        current_row_idx += 3

    plt.title(r"$\mathbb{R}$ copy states", fontsize=14, pad=20)
    plt.tight_layout()

    return fig


def plot_he(
    frame,
    channels=["image", "category"],
    base_width=4,
    base_height=4,
    max_cols=3,
):
    """
    Plot dynamically selected channels and categories for H&E images.
    """
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()

    n_channels = len(channels)
    if n_channels == 0:
        raise ValueError("The 'channels' list cannot be empty.")

    cols = min(n_channels, max_cols)
    rows = math.ceil(n_channels / cols)

    n_points = frame.shape[0]
    marker_size = np.clip(12000.0 / n_points, 0.1, 25.0)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(base_width * cols, base_height * rows),
        dpi=300,
        facecolor="white"
    )

    if n_channels == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    cmap_dict = {
        "red": "Reds",
        "green": "Greens",
        "blue": "Blues",
        "gray": "gray",
    }

    for i, col in enumerate(channels):
        ax = axes[i]
        col_lower = col.lower()

        if col_lower == "image":
            rgb = frame[["red", "green", "blue"]].values

            norm_factor = 255.0 if rgb.max() > 1.0 else 1.0
            rgb_norm = np.clip(rgb / norm_factor, 0, 1)

            ax.scatter(frame["x"], -frame["y"], c=rgb_norm, s=marker_size, linewidth=0)

        elif col_lower in ["category", "label"]:
            label_col = "label" if "label" in frame.columns else col 
            num_labels = len(np.unique(frame[label_col]))
            cmap_cat = mpl.colormaps["tab20c"].resampled(num_labels)

            sc = ax.scatter(
                frame["x"],
                -frame["y"],
                c=frame[label_col],
                s=marker_size,
                cmap=cmap_cat,
                norm=plt.Normalize(vmin=0, vmax=num_labels - 1),
                linewidth=0,
            )
            plt.colorbar(sc, ax=ax, ticks=np.arange(num_labels), fraction=0.046, pad=0.04)

        else:
            cmap = cmap_dict.get(col_lower, "viridis")
            sc = ax.scatter(
                frame["x"], -frame["y"], c=frame[col], s=marker_size, cmap=cmap, linewidth=0
            )
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Matched to plot_clones_spatial title placement and sizing
        ax.text(
            0.05,                      
            0.99,                     
            col.capitalize(), 
            transform=ax.transAxes,
            ha="left", 
            va="bottom", 
            fontsize=9,
        )
        
        # ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[n_channels:]:
        ax.axis("off")

    # fig.tight_layout()
    
    return fig
