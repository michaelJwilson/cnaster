import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from cnaster.palette import get_full_palette

NORMAL_OPACITY = 0.25

def get_intervals(pred_cnv):
    """
    Find contiguous intervals in the state label array (pred_cnv)
    where the copy number state is the same. Vectorized for performance.

    Returns a list of intervals (start index, end index) into the array
    and the corresponding array of state label for each interval.
    """
    pred_cnv = np.asarray(pred_cnv)
    if len(pred_cnv) == 0:
        return [], []

    changes = np.where(pred_cnv[:-1] != pred_cnv[1:])[0] + 1
    splits = np.concatenate(([0], changes, [len(pred_cnv)]))

    intervals = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
    labs = pred_cnv[splits[:-1]].tolist()

    return intervals, labs


def _draw_mirror_chevrons(
    ax: plt.Axes, x0: float, y_b: float, w: float, h_sub: float, direction: int
):
    """
    Mirrored events, e.g. (A, B) -> (B, A) between clones on the same segment,
    shaded by tight, vertical, mirrored chevrons according to the mirror direction.
    """
    n_chev = 2
    chev_unit = w * 0.48
    gap = w * 0.04
    total_w = n_chev * chev_unit + (n_chev - 1) * gap
    x_start = x0 + (w - total_w) / 2.0

    y_high = y_b + 2 * h_sub
    y_low = y_b

    for i in range(n_chev):
        cx_left = x_start + i * (chev_unit + gap)
        cx_right = cx_left + chev_unit
        cx_mid = (cx_left + cx_right) / 2.0

        xs = [cx_left, cx_mid, cx_right]
        ys = [y_low, y_high, y_low] if direction > 0 else [y_high, y_low, y_high]

        ax.plot(
            xs,
            ys,
            color="black",
            linewidth=1.2,
            alpha=0.5,
            solid_capstyle="round",
        )


def plot_ascn_legend(
    ax: plt.Axes,
    box_w: float = 0.8,
    box_h: float = 0.8,
    tick_len: float = 0.08,
    label_fontsize: int = 8,
    palette_name: str = "chisel_single",
):
    """Draw a horizontal color bar legend for single-allele cna values."""
    state_style, ordered_acn = get_full_palette(palette_name)
    boxes = list(ordered_acn)

    ax.axis("off")

    swatch_w = box_w * 0.5
    ax.add_patch(
        Rectangle(
            (0.0, 0.0), swatch_w, box_h, facecolor="white", edgecolor="black"
        )
    )

    _draw_mirror_chevrons(ax, 0.0, 0.0, swatch_w, box_h / 2, direction=1)

    ax.text(
        swatch_w / 2.0,
        -tick_len - 0.04,
        "Mirror",
        ha="center",
        va="top",
        fontsize=label_fontsize,
    )

    x0 = swatch_w + 0.5 

    for i, label in enumerate(boxes):
        color = state_style.get(label)
        rect = Rectangle(
            (x0 + i * box_w, 0.0),
            box_w,
            box_h,
            facecolor=mcolors.to_rgba(color, alpha=(NORMAL_OPACITY if label == 1 else 1.0)), 
            edgecolor="black", 
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
        )

    total_boxes_w = len(boxes) * box_w
    
    ax.text(
        x0 + total_boxes_w + 0.2,
        box_h / 2.0,
        r"$\mathbb{N}$" + "-CNA",
        fontsize=label_fontsize,
        ha="left",
        va="center",
    )

    # Adjust x limit to encapsulate the entire updated legend span
    ax.set_xlim(0.0, x0 + total_boxes_w + 2.0)
    ax.set_ylim(-0.5, box_h + 0.2)
    ax.set_aspect("auto")

    return ax


def plot_copy_number_profile(
    df_cnv: pd.DataFrame,
    ax: plt.Axes = None,
    height: float = 1.0,
    title: str = None,
    show_clone_name: bool = True,
    plot_chrname: bool = True,
    figsize: tuple = None,
    palette_name: str = "chisel_single",
):
    """
    Plot (allele-specific) per-clone cna profiles where horizontal width is strictly
    proportional to the number of genomic segments.
    """

    state_style, _ = get_full_palette(palette_name)

    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]

    A_full = df_cnv[[f"clone{cid} A" for cid in clone_ids]].fillna(1).to_numpy()
    B_full = df_cnv[[f"clone{cid} B" for cid in clone_ids]].fillna(1).to_numpy()

    # FIX TODO BUG: Use get_intervals(...) instead of duplicated array indexing logic
    deviations = []
    for k in range(len(clone_ids)):
        a_col, b_col = A_full[:, k], B_full[:, k]
        
        # Combine into a single state array to identify contiguous blocks
        encoded = a_col * 1_000 + b_col
        intervals, _ = get_intervals(encoded)
        
        # Extract the start indices of each interval
        starts = [s for s, e in intervals]
        
        a_rle = a_col[starts]
        b_rle = b_col[starts]
        
        # Sum the deviation of just the segments (unweighted by genomic length)
        deviations.append(np.sum(np.abs(a_rle - 1) + np.abs(b_rle - 1)))

    clone_ids = [clone_ids[i] for i in np.argsort(deviations)]

    num_clones = len(clone_ids)

    if ax is None:
        figsize = figsize or (20, max(3.0, 1.2 * num_clones))
        fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor="white")

        fig.subplots_adjust(bottom=0.15)
    else:
        fig = ax.figure

    h = height / num_clones
    clone_gap = 0.2 * h
    h_pair = h - clone_gap
    h_sub = h_pair / 2
    y_gap = clone_gap / 2

    ch_offset = 0
    ch_coords = []
    chs = []

    for ch, df_ch in df_cnv.groupby("CHR", sort=False):
        chs.append(ch)
        ch_coords.append(ch_offset)
        n_rows = len(df_ch)

        A_mat = df_ch[[f"clone{cid} A" for cid in clone_ids]].to_numpy()
        B_mat = df_ch[[f"clone{cid} B" for cid in clone_ids]].to_numpy()

        # Keep directions to dictate the orientation of the chevron (up/down)
        dirs_mat = np.zeros_like(A_mat, dtype=int)
        dirs_mat[A_mat > B_mat] = 1
        dirs_mat[A_mat < B_mat] = -1

        has_mirror = np.zeros(n_rows, dtype=bool)

        for c in range(num_clones):
            exact_swap = (
                (A_mat == B_mat[:, [c]]) & (B_mat == A_mat[:, [c]]) & (A_mat != B_mat)
            )
            has_mirror |= np.any(exact_swap, axis=1)

        for k, _ in enumerate(clone_ids):
            a_states = A_mat[:, k]
            b_states = B_mat[:, k]

            dirs = dirs_mat[:, k]

            encoded_states = a_states * 1_000 + b_states * 10 + has_mirror.astype(int)
            intervals, _ = get_intervals(encoded_states)

            k_plot = num_clones - k - 1
            y_b = y_gap + h * k_plot
            y_a = h_sub + y_b

            for s, e in intervals:
                x0 = ch_offset + s
                w = e - s

                cna, cnb = a_states[s], b_states[s]
                is_mirror, direction = has_mirror[s], dirs[s]

                ax.add_patch(
                    Rectangle(
                        (x0, y_b),
                        w,
                        h_sub,
                        facecolor=state_style.get(cnb, state_style.get("default", "lightgray")),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cnb == 1 else 1.0),
                    )
                )

                ax.add_patch(
                    Rectangle(
                        (x0, y_a),
                        w,
                        h_sub,
                        facecolor=state_style.get(cna, state_style.get("default", "lightgray")),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cna == 1 else 1.0),
                    )
                )

                if is_mirror and direction != 0:
                    _draw_mirror_chevrons(ax, x0, y_b, w, h_sub, direction)

        ch_offset += n_rows

        # TODO BUG vline at chr_offset = 0 
        for k in range(num_clones):
            y_b_k = k * h + y_gap
            ax.vlines(
                ch_offset,
                ymin=y_b_k,
                ymax=y_b_k + h_pair,
                linewidth=1,
                colors="black",
                clip_on=False,
            )

    ch_coords.append(ch_offset)

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
                    linewidth=1,
                )
            )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_chrname and chs:
        midpoints = [ch_coords[i] for i in range(len(ch_coords) - 1)]
        ax.set_xticks(midpoints)

        chr_labels = [f"chr{ch}" if str(ch).isdigit() else str(ch) for ch in chs]

        ax.set_xticklabels(chr_labels, rotation=45, fontsize=8, ha="left")
        ax.tick_params(
            axis="x",
            labeltop=False,
            labelbottom=True,
            top=False,
            bottom=False,
            pad=-5,
        )
    else:
        ax.set_xticks([])

    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ylabels = [
        f"Clone {cid}" if show_clone_name else str(cid) for cid in reversed(clone_ids)
    ]
    ax.set_yticklabels(ylabels, fontsize=8, va="center", ha="left", rotation=90)

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

    legend_ax = fig.add_axes([0.15, 0.02, 0.7, 0.05])

    plot_ascn_legend(legend_ax, palette_name=palette_name)

    if title:
        ax.set_title(title)

    return fig