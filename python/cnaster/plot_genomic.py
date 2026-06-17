import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
    """Standardizes the styling for RDR and BAF genomic tracks."""
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])
    ax.set_xlim([0, n_obs])

    if remove_xticks:
        ax.set_xticks([])

    # Draw horizontal guide lines
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
    ax, clone_label, n_spots, n_umis, n_snp_umis, tumor_prop=None
):
    """Annotates the axis with clone identity and spot/UMI statistics."""
    ax.text(
        -0.04,
        0.5,
        cast_clone_label(str(clone_label)),
        ha="center",
        va="center",
        fontsize=12,
        rotation="vertical",
        transform=ax.transAxes,
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
        )

        # HMM Model Fits
        if res is not None:
            max_pred = np.argmax(res["log_gamma"], axis=0)
            this_pred = max_pred[(s * n_obs) : (s * n_obs + n_obs)] % res["n_states"]
            segments, labs = get_intervals(this_pred)
            mus = np.exp(res["new_log_mu"])
            ps = res["new_p_binom"]

            for i, (seg, state) in enumerate(zip(segments, labs)):
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
    df_cnv: pd.DataFrame,
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

    chisel_palette, ordered_acn = get_full_palette(palette_name)
    map_cn = {x: i for i, x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]

    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])
    if "0" not in final_clone_ids:
        final_clone_ids = np.array(["0"] + list(final_clone_ids))

    n_states = res_combine["new_p_binom"].shape[0]
    unique_chrs = np.unique(df_cnv.CHR.values)

    assert single_X.shape[0] == df_cnv.shape[0], "Genomic segment counts mismatch."

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

    fig, axes = _create_clone_gridspec(
        len(nonempty_clones), 2, base_height, sample_list
    )

    for s, c in enumerate(nonempty_clones):
        cid = final_clone_ids[c]
        ax_rdr, ax_baf = axes[2 * s], axes[2 * s + 1]

        # Determine states for color mapping
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
        else:
            hue = pd.Categorical(
                res_combine["pred_cnv"][:, c],
                categories=np.arange(n_states),
                ordered=True,
            )
            palette = palette

        x_vals = np.arange(n_obs)
        point_colors = [
            {i: palette[i] for i in range(len(palette))}[h] for h in hue.codes
        ]

        # --- Plot RDR ---
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
                ecolor=point_colors,
                elinewidth=0.5,
                alpha=0.75,
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
        _format_track_axis(
            ax_rdr,
            "\nRDR",
            [-0.5, rdr_ylim],
            np.arange(0, rdr_ylim + 1.0, 1.0),
            remove_xticks,
            n_obs,
        )

        # --- Plot BAF ---
        baf_vals = X[:, 1, c] / total_bb_RD[:, c]
        if plot_baf_errors is not None:
            n_counts = np.maximum(total_bb_RD[:, c], 1)  # Prevent division by zero

            if plot_baf_errors == "wald":
                std_err_baf = np.sqrt(baf_vals * (1 - baf_vals) / n_counts)
            elif plot_baf_errors == "beta":
                k, n = X[:, 1, c], total_bb_RD[:, c]
                alpha, beta = k + 1, n - k + 1
                alpha_beta_sum = alpha + beta
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
                alpha=0.75,
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

        # --- Draw Fits and Annotations ---
        for i, seg in enumerate(segments):
            ax_rdr.plot(
                seg,
                [np.exp(res_combine["new_log_mu"][labs[i], c])] * 2,
                c="k",
                linewidth=0.5,
                zorder=2,
            )
            ax_baf.plot(
                seg,
                [res_combine["new_p_binom"][labs[i], c]] * 2,
                c="k",
                linewidth=0.5,
                zorder=2,
            )
            ax_baf.plot(
                seg,
                [1.0 - res_combine["new_p_binom"][labs[i], c]] * 2,
                c="k",
                linewidth=0.5,
                linestyle="--",
                zorder=2,
            )

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

        _annotate_clone_stats(
            ax_rdr,
            cid,
            spots_per_clone[c],
            np.sum(X[:, 0, c]),
            np.sum(total_bb_RD[:, c]),
        )

    _draw_chromosome_boundaries(axes, lengths, unique_chrs, chrtext_shift)
    fig.tight_layout()

    return fig


def _draw_mirrored_loh_chevrons(
    ax: plt.Axes, x0: float, y_b: float, w: float, h_sub: float, direction: int
):
    """Private helper to draw the ≪ / ≫ chevrons for Mirrored LOH."""
    n_chev = 2
    chev_unit = w * 0.12
    gap = w * 0.04
    total_w = n_chev * chev_unit + (n_chev - 1) * gap
    x_start = x0 + (w - total_w) / 2.0

    y_high = y_b + 2 * h_sub
    y_low = y_b
    y_mid = (y_high + y_low) / 2.0

    for i in range(n_chev):
        cx_left = x_start + i * (chev_unit + gap)
        cx_right = cx_left + chev_unit

        xs = (
            [cx_left, cx_right, cx_left]
            if direction > 0
            else [cx_right, cx_left, cx_right]
        )

        ax.plot(
            xs,
            [y_high, y_mid, y_low],
            color="black",
            linewidth=1.2,
            alpha=0.5,
            solid_capstyle="round",
            transform=ax.get_xaxis_transform(),
        )


def plot_ascn_profile(
    ax: plt.Axes,
    bin_info: pd.DataFrame,
    regions: pd.DataFrame,
    width: float = 20.0,
    height: float = 1.0,
    title: str = None,
    ylabel: str = None,
    plot_chrname: bool = True,
    show_prop: bool = True,
    show_clone_name: bool = True,
    clone_ploidies: dict = None,
):
    """
    Plot allele-specific CN profile with two sub-bars (A/B) per clone.

    Expects `bin_info` to contain a 'CNP' column with semicolon-separated,
    pipe-delimited string states (e.g., "prefix;2|1;1|1").
    """
    state_style, _ = get_full_palette()

    # Extract baseline dimensions
    first_cnp = str(bin_info["CNP"].iloc[0]).split(";")
    num_clones = len(first_cnp) - 1

    h = height / num_clones
    clone_gap = 0.10 * h
    h_pair = h - clone_gap
    h_sub = h_pair / 2
    y_gap = clone_gap / 2

    bulk_props = np.array([float(v) for v in str(bin_info["PROPS"].iloc[0]).split(";")])

    regions_chs = regions.groupby(by="#CHR", sort=False)
    bins_chs = bin_info.groupby(by="#CHR", sort=False, observed=True)

    ch_offset = 0
    ch_coords = []
    chs = bin_info["#CHR"].unique()

    for ch in chs:
        ch_coords.append(ch_offset)
        regions_ch = regions_chs.get_group(ch)
        bins_ch = bins_chs.get_group(ch)

        for wl_segment in regions_ch.itertuples():
            wl_start = wl_segment.START
            wl_end = wl_segment.END
            seg_end = ch_offset + (wl_end - wl_start)

            bins_seg = bins_ch.loc[
                (bins_ch["START"] >= wl_start) & (bins_ch["END"] <= wl_end)
            ]

            if bins_seg.empty:
                ch_offset = seg_end
                continue

            # Vectorized offset calculations
            bin_starts = (bins_seg["START"] - wl_start + ch_offset).to_numpy()
            bin_ends = (bins_seg["END"] - wl_start + ch_offset).to_numpy()

            # Pre-parse the heavy strings BEFORE the rendering loop
            # Extracts list of (A, B) integer tuples per bin
            parsed_cnps = [
                [
                    (int(cn.split("|")[0]), int(cn.split("|")[1]))
                    for cn in str(val).split(";")[1:]
                ]
                for val in bins_seg["CNP"]
            ]

            ch_offset = seg_end

            # Optimized rendering loop
            for bi in range(len(bins_seg)):
                x0, bin_end = bin_starts[bi], bin_ends[bi]
                w = bin_end - x0
                clone_states = parsed_cnps[bi]

                any_non_loh = any(a > 0 and b > 0 for a, b in clone_states)
                dirs = [
                    (1 if (a > 0 and b == 0) else (-1 if (a == 0 and b > 0) else 0))
                    for a, b in clone_states
                ]
                has_mirror = (not any_non_loh) and (1 in dirs) and (-1 in dirs)

                for k in range(num_clones):
                    cna, cnb = clone_states[num_clones - k - 1]
                    direction = dirs[num_clones - k - 1]
                    y_b = k * h + y_gap
                    y_a = y_b + h_sub

                    # B Allele
                    ax.add_patch(
                        Rectangle(
                            (x0, y_b),
                            w,
                            h_sub,
                            facecolor=state_style.get(
                                (cna, cnb), state_style["default"]
                            ),
                            edgecolor="none",
                            transform=ax.get_xaxis_transform(),
                            linewidth=0,
                            alpha=1.0 if cnb == 0 else 0.5,
                        )
                    )

                    # A Allele
                    ax.add_patch(
                        Rectangle(
                            (x0, y_a),
                            w,
                            h_sub,
                            facecolor=state_style.get(
                                (cna, cnb), state_style["default"]
                            ),
                            edgecolor="none",
                            transform=ax.get_xaxis_transform(),
                            linewidth=0,
                            alpha=1.0 if cna == 0 else 0.5,
                        )
                    )

                    if has_mirror and direction != 0:
                        _draw_mirrored_loh_chevrons(ax, x0, y_b, w, h_sub, direction)

            # Centromere boundary lines
            if wl_segment.Index != regions_ch.index[-1]:
                for k in range(num_clones):
                    ax.vlines(
                        ch_offset,
                        ymin=k * h + y_gap,
                        ymax=k * h + y_gap + h_pair,
                        transform=ax.get_xaxis_transform(),
                        linewidth=0.5,
                        colors="black",
                        linestyles="dashed",
                    )

        if ch != chs[-1]:
            line = ax.vlines(
                ch_offset,
                ymin=0,
                ymax=1.15,
                transform=ax.get_xaxis_transform(),
                linewidth=1,
                colors="black",
            )
            line.set_clip_on(False)

    ch_coords.append(ch_offset)

    # UI Formatting
    for k in range(num_clones):
        y_b_k = k * h + y_gap
        for y0 in (y_b_k, y_b_k + h_sub):
            ax.add_patch(
                Rectangle(
                    (0, y0),
                    ch_offset,
                    h_sub,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.5,
                    transform=ax.get_xaxis_transform(),
                )
            )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_chrname:
        ax.set_xticks(
            [
                ch_coords[i] + (ch_coords[i + 1] - ch_coords[i]) // 2
                for i in range(len(ch_coords) - 1)
            ]
        )
        ax.set_xticklabels(chs, rotation=60, fontsize=8)
        ax.tick_params(
            axis="x", labeltop=True, labelbottom=False, top=False, bottom=False
        )
    else:
        ax.set_xticks([])

    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ylabels = []
    for ci in range(num_clones, 0, -1):
        prop = round(bulk_props[ci] * 100, 1)
        lines = [f"Clone {ci}" if show_clone_name else str(ci)]
        if clone_ploidies and f"clone{ci}" in clone_ploidies:
            lines.append(f"ploidy {round(clone_ploidies[f'clone{ci}'], 2)}")
        if show_prop:
            lines.append(f"prop {prop}%")
        ylabels.append("\n".join(lines))

    ax.set_yticklabels(ylabels, fontsize=8, va="center")

    minor_positions, minor_labels = [], []
    for k in range(num_clones):
        minor_positions.extend(
            [k * h + y_gap + h_sub * 0.5, k * h + y_gap + h_sub * 1.5]
        )
        minor_labels.extend(["B", "A"])

    ax.set_yticks(minor_positions, minor=True)
    ax.set_yticklabels(minor_labels, minor=True, fontsize=6)
    ax.tick_params(axis="y", which="minor", left=False, right=False, pad=2)

    ax.set_ylim(0, num_clones * h)
    ax.tick_params(axis="y", which="major", left=True, right=False, length=4, pad=20)

    if ylabel:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center")
    if title:
        ax.set_title(title)

    return ax


def plot_ascn_legend(
    ax: plt.Axes,
    box_w: float = 1.2,
    box_h: float = 0.4,
    tick_len: float = 0.08,
    label_fontsize: int = 12,
):
    """Draw a horizontal color bar legend for allele CN values."""
    state_style, ordered_acn = get_full_palette()
    boxes = list(ordered_acn) + ["7+"]
    ax.axis("off")
    x0 = 0.0

    for i, label in enumerate(boxes):
        color = state_style["default"] if label == "7+" else state_style[label]
        rect = Rectangle(
            (x0 + i * box_w, 0.0),
            box_w,
            box_h,
            facecolor=color,
            edgecolor="black",
            alpha=1.0 if label == 0 else 0.5,
        )
        ax.add_patch(rect)
        xc = x0 + i * box_w + box_w / 2.0
        ax.plot([xc, xc], [-tick_len, 0.0], color="black", linewidth=0.8)
        ax.text(
            xc,
            -tick_len - 0.04,
            str(label),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
        )

    total_w = len(boxes) * box_w
    ax.text(
        -0.3,
        box_h / 2.0,
        "Allele copy number",
        fontsize=label_fontsize,
        fontweight="bold",
        ha="right",
        va="center",
    )

    swatch_w = box_w * 0.7
    chev_box_x = total_w + 1.0
    ax.add_patch(
        Rectangle(
            (chev_box_x, 0.0), swatch_w, box_h, facecolor="white", edgecolor="black"
        )
    )

    _draw_mirrored_loh_chevrons(ax, chev_box_x, 0.0, swatch_w, box_h / 2, direction=1)

    ax.text(
        chev_box_x + swatch_w / 2.0,
        -tick_len - 0.04,
        "Mirrored LOH",
        ha="center",
        va="top",
        fontsize=label_fontsize,
        fontweight="bold",
    )

    ax.set_xlim(-2.0, chev_box_x + swatch_w + 0.5)
    ax.set_ylim(-0.5, box_h + 0.2)
    ax.set_aspect("auto")

    return ax
