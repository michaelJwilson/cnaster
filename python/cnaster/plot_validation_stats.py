import os
import glob
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_rows', None)

def load_validation_stats(stats_dir):
    stats_path = Path(stats_dir)
    rows = []
    for ypath in stats_path.glob("validation_stats_*.yaml"):
        with open(ypath, "r") as f:
            data = yaml.safe_load(f)
        
        row = {
            "sample_id": data.get("sample_id"),
            "initialization": data.get("initialization"),
            "loglike": data.get("loglike"),
            "num_clones": data.get("num_clones"),
            "normal_rate": data.get("normal_rate"),
            "match_rate": data.get("match_rate"),
            "correct_rate": data.get("correct_rate"),
            "ari": data.get("ari"),
            "clone_mapping_success_rate": data.get("clone_mapping_success_rate"),
            "normal_recovery_rate": data.get("normal_recovery_rate"),
            "cna_recovery_rate": data.get("cna_recovery_rate"),
            "cna_false_positive_rate": data.get("cna_false_positive_rate"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)

    # Parse parameters from sample_id like:
    # numcnas1.2_cnasize1e7_ploidy2_random2
    pat = r"numcnas(?P<numcnas>[\d\.]+)_cnasize(?P<cnasize>[\deE\+\-\.]+)_ploidy(?P<ploidy>[\d\.]+)_random(?P<random>\d+)"
    extracted = df["sample_id"].astype(str).str.extract(pat)

    for c in ["numcnas", "cnasize", "ploidy"]:
        extracted[c] = pd.to_numeric(extracted[c], errors="coerce")
    extracted["random"] = pd.to_numeric(extracted["random"], downcast="integer", errors="coerce")

    # Group label including ploidy (remove only random)
    group = df["sample_id"].astype(str).str.replace(r"_random\d+$", "", regex=True)

    df = pd.concat([df, extracted, group.rename("group")], axis=1)
    return df


def plot_metrics(df, method, output_dir=None):
    sns.set_context("paper", font_scale=0.9)
    sns.set_style("ticks")

    n_samples = df["sample_id"].nunique()

    # Reorder groups: by cnasize (largest first) then sum of numcnas components (desc)
    def _numcnas_sum(x):
        parts = str(x).split(".")
        if len(parts) > 1 and all(p.isdigit() for p in parts):
            return sum(int(p) for p in parts)
        try:
            return float(x)
        except Exception:
            return np.nan
    order_df = (
        df[["group", "numcnas", "cnasize"]]
        .dropna()
        .drop_duplicates()
        .assign(numcnas_sum=lambda d: d["numcnas"].apply(_numcnas_sum))
        .sort_values(["cnasize", "numcnas_sum"], ascending=[False, False])
    )
    group_order = order_df["group"].tolist()

    # Build short, readable x tick labels
    def _compact_size(x):
        if pd.isna(x):
            return "NA"
        try:
            x = float(x)
        except Exception:
            return str(x)
        if x > 0:
            exp = int(np.round(np.log10(x)))
            if np.isclose(x, 10 ** exp, rtol=1e-8, atol=0):
                return f"1e{exp}"
        s = np.format_float_scientific(x, precision=0, exp_digits=1)
        return s.replace("+0", "").replace("+", "")
    
    label_map = {
        r.group: f"({str(r.numcnas).replace('.',',')}, {_compact_size(r.cnasize)})"
        for r in order_df.itertuples(index=False)
    }

    # Metrics including loglike
    melt_cols = [
        "loglike",
        "normal_rate",
        "cna_recovery_rate",
        "normal_recovery_rate",
        "clone_mapping_success_rate",
        "ari",
    ]
    metric_labels = {
        "loglike": "Log-likelihood",
        "normal_rate": "(1,1) rate",
        "cna_recovery_rate": "$\mathbb{N}$-CNA recovery rate",
        "normal_recovery_rate": "(1,1) recovery rate",
        "clone_mapping_success_rate": "Clone recovery rate",
        "ari": "ARI",
    }

    df_melt = df.melt(
        id_vars=["group", "random"],
        value_vars=melt_cols,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["group"])

    # Layout: 3x2 grid
    layout = [
        ("loglike", (0, 0)),
        ("normal_rate", (0, 1)),
        ("cna_recovery_rate", (1, 0)),
        ("normal_recovery_rate", (1, 1)),
        ("clone_mapping_success_rate", (2, 0)),
        ("ari", (2, 1)),
    ]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9), squeeze=True)
    fig.suptitle(rf"$\tt{{{method}}}$ — {n_samples} samples", fontsize=14, y=0.98)

    used_axes = set()
    for metric, (r, c) in layout:
        ax = axes[r, c]
        used_axes.add((r, c))
        data_m = df_melt[df_melt["metric"] == metric]
        
        # Filter to only groups that have at least one non-NaN value for this metric
        data_m_valid = data_m.dropna(subset=["value"])
        present_groups = [g for g in group_order if g in set(data_m_valid["group"])]
        
        # If no data at all, skip this panel
        if not present_groups:
            ax.set_visible(False)
            continue
        
        sns.boxplot(
            data=data_m_valid,
            x="group",
            y="value",
            order=present_groups,
            ax=ax,
            color="#4A90E2",  # blue color
            showcaps=True,
            showfliers=False,
            boxprops={"alpha": 0.4},
            whiskerprops={"color": "#2E5A8F", "linewidth": 1},
            medianprops={"color": "black", "linewidth": 1.2},
        )
        sns.stripplot(
            data=data_m_valid,
            x="group",
            y="value",
            order=present_groups,
            ax=ax,
            color="black",
            alpha=0.7,
            size=3,
            jitter=0.1,
            marker="o",
            edgecolor="black",
            linewidth=0.2,
        )
        ax.grid(False)
        ax.set_ylabel(metric_labels.get(metric, metric))
        
        ax.set_ylim(-0.05, 1.05)
        
        # Bottom row: axis label + tick labels; other rows: hide tick labels
        if r == nrows - 1:
            ax.set_xlabel(r"CNA realization type")
            ax.set_xticklabels(
                [label_map.get(g, g) for g in present_groups],
                rotation=45,
                ha="right",
            )
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    # Hide unused axes
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used_axes:
                axes[r, c].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)

    out_path = Path(".") / f"{method}_validation.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)

def main():
    method = "calicost"
    # method = "cnaster"
    
    stats_dir = f"/Users/mw9568/Work/ragr/sim/stats/{method}"
    df = load_validation_stats(stats_dir)
    
    plot_metrics(df, method, output_dir=stats_dir)

    df_rank = df.copy()
    df_rank["__loglike__"] = pd.to_numeric(df_rank["loglike"], errors="coerce").fillna(-np.inf)
    idx = df_rank.groupby("sample_id")["__loglike__"].idxmax()
    df_best = df_rank.loc[idx].drop(columns=["__loglike__"]).reset_index(drop=True)
    
    df_best = df_best.sort_values(["cna_recovery_rate"], ascending=[True])
    
    # Drop columns before printing
    cols_to_drop = ["numcnas", "cnasize", "ploidy", "random", "group"]
    df_print = df_best.drop(columns=cols_to_drop, errors='ignore')
    
    print(df_print)

if __name__ == "__main__":
    main()
