import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from cnaster.palette import get_full_palette


def get_intervals(pred_cnv):
    """
    Find contiguous intervals in the state label array (pred_cnv)
    --- typically real copy states (Z) or integer (A,B) states ---
    where the copy number state is the same.

    Returns a list of intervals (start index, end index) into the array
    and the corresponding array of state label for each interval.
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
            # NB next label switch
            t = t[0]

            # NB add the interval (run start index to run end index)
            intervals.append((s, s + t))

            # NB add the corresponding state label for this new interval.
            labs.append(pred_cnv[s])

            # NB update the index.
            s = s + t

    return intervals, labs


def _draw_mirrored_loh_chevrons(
    ax: plt.Axes, x0: float, y_b: float, w: float, h_sub: float, direction: int
):
    """
    Mirrored events, e.g. LOH, between individual sub-clones should have 
    their colored segments (A and B) marked by tight, vertical, mirrored chevrons.
    """
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


def plot_ascn_profile(
    ax: plt.Axes,
    df_cnv: pd.DataFrame,
    height: float = 1.0,
    title: str = None,
    show_clone_name: bool = True,
):
    """
    Plot (allele-specific) per-clone CNA profiles where width is strictly
    proportional to the number of segments using the legacy 1D get_intervals.

    df_cnv contains segment/bin level:
        CHR, clone{cid} Z, clone{cid} A, clone{cid} B
    """
    state_style, _ = get_full_palette()

    # Extract clone IDs from column names
    a_cols = [c for c in df_cnv.columns if c.endswith(" A")]
    clone_ids = [c.split(" ")[0][5:] for c in a_cols]
    num_clones = len(clone_ids)

    # Dimensional setup
    h = height / num_clones
    clone_gap = 0.10 * h
    h_pair = h - clone_gap
    h_sub = h_pair / 2
    y_gap = clone_gap / 2

    # Prepare groupby object
    df_chs = df_cnv.groupby(by="CHR", sort=False)

    ch_offset = 0
    ch_coords = []
    chs = df_cnv["CHR"].unique()

    for ch in chs:
        ch_coords.append(ch_offset)
        
        if ch not in df_chs.groups:
            continue
            
        df_ch = df_chs.get_group(ch)
        n_rows = len(df_ch)

        # Extract integer copy numbers into matrices
        A_mat = df_ch[[f"clone{cid} A" for cid in clone_ids]].to_numpy()
        B_mat = df_ch[[f"clone{cid} B" for cid in clone_ids]].to_numpy()
        
        # Fast check for mirrored LOH across all clones per row
        any_non_loh = np.any((A_mat > 0) & (B_mat > 0), axis=1)
        
        # Direction matrix: 1 if A>0/B=0, -1 if A=0/B>0, else 0
        dirs_mat = np.where((A_mat > 0) & (B_mat == 0), 1, 
                   np.where((A_mat == 0) & (B_mat > 0), -1, 0))
        
        # has_mirror is true if no clone has non-loh, and both directions exist
        has_mirror = (~any_non_loh) & np.any(dirs_mat == 1, axis=1) & np.any(dirs_mat == -1, axis=1)

        for k, cid in enumerate(clone_ids):
            a_states = A_mat[:, k]
            b_states = B_mat[:, k]
            dirs = dirs_mat[:, k]

            # Fast numeric encoding to feed into the 1D get_intervals array 
            # to guarantee breaks happen when A, B, or the cross-clone mirror state changes.
            encoded_states = a_states * 1000 + b_states * 10 + has_mirror.astype(int)
            
            intervals, _ = get_intervals(encoded_states)

            k_plot = num_clones - k - 1
            y_b = k_plot * h + y_gap
            y_a = y_b + h_sub

            for (s, e) in intervals:
                # Map segment counts directly to width and x0
                x0 = ch_offset + s
                w = e - s
                
                # Retrieve actual states for this interval
                cna = a_states[s]
                cnb = b_states[s]
                is_mirror = has_mirror[s]
                direction = dirs[s]

                # B Allele (Bottom sub-bar)
                ax.add_patch(
                    Rectangle(
                        (x0, y_b), w, h_sub,
                        facecolor=state_style.get((cna, cnb), state_style["default"]),
                        edgecolor="none",
                        transform=ax.get_xaxis_transform(),
                        linewidth=0,
                        alpha=1.0 if cnb == 0 else 0.5,
                    )
                )

                # A Allele (Top sub-bar)
                ax.add_patch(
                    Rectangle(
                        (x0, y_a), w, h_sub,
                        facecolor=state_style.get((cna, cnb), state_style["default"]),
                        edgecolor="none",
                        transform=ax.get_xaxis_transform(),
                        linewidth=0,
                        alpha=1.0 if cna == 0 else 0.5,
                    )
                )

                if is_mirror and direction != 0:
                    _draw_mirrored_loh_chevrons(ax, x0, y_b, w, h_sub, direction)

        # Advance offset by the exact number of segments (rows) in this chromosome
        ch_offset += n_rows

        # Chromosome dividing line
        if ch != chs[-1]:
            line = ax.vlines(
                ch_offset, ymin=0, ymax=1.15,
                transform=ax.get_xaxis_transform(),
                linewidth=1, colors="black",
            )
            line.set_clip_on(False)

    ch_coords.append(ch_offset)

    # UI Formatting
    for k in range(num_clones):
        y_b_k = k * h + y_gap
        for y0 in (y_b_k, y_b_k + h_sub):
            ax.add_patch(
                Rectangle(
                    (0, y0), ch_offset, h_sub,
                    facecolor="none", edgecolor="black",
                    linewidth=0.5, transform=ax.get_xaxis_transform(),
                )
            )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Deprecated plot_chrname, hiding ticks
    ax.set_xticks([])

    # Generate simplified Y-axis labels
    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ylabels = [f"Clone {cid}" if show_clone_name else str(cid) for cid in reversed(clone_ids)]
    ax.set_yticklabels(ylabels, fontsize=8, va="center")

    minor_positions, minor_labels = [], []
    for k in range(num_clones):
        minor_positions.extend([k * h + y_gap + h_sub * 0.5, k * h + y_gap + h_sub * 1.5])
        minor_labels.extend(["B", "A"])

    ax.set_yticks(minor_positions, minor=True)
    ax.set_yticklabels(minor_labels, minor=True, fontsize=6)
    ax.tick_params(axis="y", which="minor", left=False, right=False, pad=2)

    ax.set_ylim(0, num_clones * h)
    ax.tick_params(axis="y", which="major", left=True, right=False, length=4, pad=20)

    if title:
        ax.set_title(title)

    return ax