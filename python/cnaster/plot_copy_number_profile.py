import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from cnaster.palette import get_full_palette


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
