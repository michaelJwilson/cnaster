import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import logging
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

logger = logging.getLogger(__name__)


def get_full_palette():
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
    ordered_acn = [
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
    return palette, ordered_acn


def get_intervals(pred_cnv):
    intervals, labs = [],[]
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


def plot_rdr_baf(
    df_cnv,
    lengths,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res_combine,
    single_tumor_prop=None,
    clone_ids=None,
    remove_xticks=True,
    rdr_ylim=5,
    chrtext_shift=-0.3,
    base_height=3.2,
    pointsize=15,
    linewidth=1,
    palette="chisel",
):
    logger.info(f"Plotting final rdr+baf.")
    
    chisel_palette, ordered_acn = get_full_palette()
    map_cn = {x: i for i, x in enumerate(ordered_acn)}
    colors = [chisel_palette[c] for c in ordered_acn]
    
    final_clone_ids = np.unique([x.split(" ")[0][5:] for x in df_cnv.columns[3:]])
    if "0" not in final_clone_ids:
        final_clone_ids = np.array(["0"] + list(final_clone_ids))
    assert (clone_ids is None) or np.all(
        [(cid in final_clone_ids) for cid in clone_ids]
    )
    unique_chrs = np.unique(df_cnv.CHR.values)

    n_states = res_combine["new_p_binom"].shape[0]

    assert single_X.shape[0] == df_cnv.shape[0]

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
    nonempty_clones = np.where(np.sum(total_bb_RD, axis=0) > 0)[0]

    assert clone_ids is None

    fig, axes = plt.subplots(
        2 * len(nonempty_clones),
        1,
        figsize=(20, base_height * len(nonempty_clones)),
        dpi=200,
        facecolor="white",
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

        if palette == "chisel":
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

        # NB plot RDR
        sns.scatterplot(
            x=np.arange(X[:, 1, c].shape[0]),
            y=X[:, 0, c] / base_nb_mean[:, c],
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="black",
            linewidth=linewidth,
            alpha=1,
            legend=False,
            ax=axes[2 * s],
        )

        axes[2 * s].set_ylabel(f"clone {cid}\nRDR")
        axes[2 * s].set_yticks(np.arange(1, rdr_ylim, 1))
        axes[2 * s].set_ylim([0, rdr_ylim])
        axes[2 * s].set_xlim([0, n_obs])
        
        if remove_xticks:
            axes[2 * s].set_xticks([])

        if palette == "chisel":
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
            x=np.arange(X[:, 1, c].shape[0]),
            y=X[:, 1, c] / total_bb_RD[:, c],
            hue=hue,
            palette=palette,
            s=pointsize,
            edgecolor="black",
            alpha=0.8,
            legend=False,
            ax=axes[2 * s + 1],
        )

        axes[2 * s + 1].set_ylabel(f"clone {cid}\nphased AF")
        axes[2 * s + 1].set_ylim([-0.1, 1.1])
        axes[2 * s + 1].set_yticks([0, 0.5, 1])
        axes[2 * s + 1].set_xlim([0, n_obs])
        if remove_xticks:
            axes[2 * s + 1].set_xticks([])
        for i, seg in enumerate(segments):
            axes[2 * s].plot(
                seg,
                [
                    np.exp(res_combine["new_log_mu"][labs[i], c]),
                    np.exp(res_combine["new_log_mu"][labs[i], c]),
                ],
                c="black",
                linewidth=2,
            )
            axes[2 * s + 1].plot(
                seg,
                [
                    res_combine["new_p_binom"][labs[i], c],
                    res_combine["new_p_binom"][labs[i], c],
                ],
                c="black",
                linewidth=2,
            )
            axes[2 * s + 1].plot(
                seg,
                [
                    1. - res_combine["new_p_binom"][labs[i], c],
                    1. - res_combine["new_p_binom"][labs[i], c],
                ],
                c="black",
                linewidth=2,
            )

    for i in range(len(lengths)):
        median_len = (
            np.sum(lengths[:(i)]) * 0.55 + np.sum(lengths[: (i + 1)]) * 0.45
        )
        axes[-1].text(
            median_len - 5,
            chrtext_shift,
            unique_chrs[i],
            transform=axes[-1].get_xaxis_transform(),
        )
        for k in range(2 * len(nonempty_clones)):
            axes[k].axvline(x=np.sum(lengths[:(i)]), c="grey", linewidth=1)
            
    fig.tight_layout()

    return fig
