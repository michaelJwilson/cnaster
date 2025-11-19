import yaml
import glob
import time
import logging
import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from pprint import pformat
from pathlib import Path
from collections import Counter
from sklearn.metrics import adjusted_rand_score

pl.Config.set_tbl_cols(-1)

start_time = time.time()


class RuntimeFormatter(logging.Formatter):
    def format(self, record):
        runtime_minutes = (time.time() - start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        return super().format(record)


formatter = RuntimeFormatter(
    fmt="%(asctime)s - %(runtime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# TODO HACK
def get_gene_ranges_path():
    # return "/u/mw9568/research/data/references/hgTables_hg38_gencode.txt"
    return "/Users/mw9568/repos/calicost/GRCh38_resources/hgTables_hg38_gencode.txt"

def read_gene_ranges():
    """
    Read in (contig, start, end) table for (coding) gene definition.
    """
    gene_ranges_path = get_gene_ranges_path()

    gene_ranges = pd.read_csv(gene_ranges_path, sep="\t", header=0, index_col=0)
    gene_ranges = gene_ranges[gene_ranges.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    # NB add chr column as integer without "chr" prefix
    gene_ranges["chr"] = [int(x[3:]) for x in gene_ranges.chrom]
    gene_ranges = gene_ranges.rename(
        columns={
            "chr": "Chromosome",
            "cdsStart": "Start",
            "cdsEnd": "End",
            "name2": "Gene",
        }
    ).reset_index()

    return pr.PyRanges(gene_ranges[["Chromosome", "Start", "End", "Gene"]])


def remap_clone_num(entries):
    """
    Remap an iterable of labels containing a clone id to a standard one, e.g.
    normal being clone 0 and other clones labelled accordingly.
    """
    entries = np.unique(entries)

    logger.info(f"Found unique entries: {entries}")

    has_normal = np.any(["normal" in xx for xx in entries])
    clone_zp = 1 if has_normal else 0

    new_entries = {}

    if has_normal:
        logger.warning("Detected 'normal' in clone labelling, assuming clone 0.")

    for entry in entries:
        orig_entry = entry
        entry = entry.replace("_copy", "")

        if entry in new_entries:
            continue

        if entry.startswith("normal"):
            new_entries[orig_entry] = entry.replace("normal", "clone_0")

        elif entry.startswith("clone"):
            parts = entry.split("_")

            clone_num = int(parts[1])
            new_clone_num = clone_num + clone_zp

            new_entries[orig_entry] = entry.replace(
                f"clone_{clone_num}", f"clone_{new_clone_num}"
            )

    logger.info(
        f"Found entry mapper={new_entries} with has_normal={has_normal} and clone_zp={clone_zp}"
    )

    return new_entries


def best_permutation_accuracy(true_labels, pred_labels):
    """
    Assuming estimated clone labels are a permutation (or noisy permutation) of the true labels,
    produce a mapping from each predicted label -> a true label (allowing multiple predicted
    labels to map to the same true label). Return that mapping, the fraction correct under
    this mapping, and the mapped prediction vector.

    NaNs in either vector are ignored pairwise.
    """
    t = np.asarray(true_labels)
    p = np.asarray(pred_labels)

    # only consider finite pairs
    mask = np.isfinite(t) & np.isfinite(p)
    t_valid = t[mask]
    p_valid = p[mask]

    if len(t_valid) == 0:
        return {}, 0.0, p.copy()

    true_uniques, true_idx = np.unique(t_valid, return_inverse=True)
    pred_uniques, pred_idx = np.unique(p_valid, return_inverse=True)

    n_true = len(true_uniques)
    n_pred = len(pred_uniques)
    n = max(n_true, n_pred)

    # contingency counts (true x pred)
    counts = np.zeros((n_true, n_pred), dtype=int)
    for ti, pi in zip(true_idx, pred_idx):
        counts[ti, pi] += 1

    # Build a mapping for every predicted label: map each pred -> true with highest counts
    many_to_one_mapping = {}
    for j, pred_val in enumerate(pred_uniques):
        # pick true row with max support for this predicted column
        best_true_idx = int(np.argmax(counts[:, j])) if n_true > 0 else 0
        many_to_one_mapping[pred_val] = true_uniques[best_true_idx]

    # Apply mapping to full pred array (preserve NaNs)
    mapped = p.copy()
    for i, val in enumerate(p):
        if not np.isfinite(val):
            continue
        mapped[i] = many_to_one_mapping.get(val, val)

    # fraction correct on finite pairs
    frac_correct = (mapped[mask] == t_valid).mean()

    return many_to_one_mapping, float(frac_correct), mapped


def get_sample_truth(root, sample_id, cna_only=False):
    logger.info(
        f"Solving for {root}/simulated_data_related/{sample_id}/truth_clone_labels.tsv"
    )

    # NB
    #         labels  x       y
    # spot_0  clone_2 0       0
    truth_clones = pd.read_csv(
        f"{root}/simulated_data_related/{sample_id}/truth_clone_labels.tsv",
        sep="\t",
        names=["barcode", "true_clone", "x", "y"],
        skiprows=1,
    )

    truth_clones["true_clone"] = truth_clones["true_clone"].map(
        remap_clone_num(truth_clones["true_clone"])
    )

    spots = truth_clones["barcode"].unique()
    spot_to_clone = dict(zip(truth_clones["barcode"], truth_clones["true_clone"]))

    # NB
    # clone	    chr	    start	    end	        A_copy	B_copy
    # clone_0	20	    51816053	61816053	0	    1
    fname = "truth_acn_profile.tsv"

    # NB clone CNAs only.
    # fname = "truth_cna.tsv"

    truth = pd.read_csv(
        f"{root}/simulated_data_related/{sample_id}/{fname}",
        sep="\t",
    ).rename(
        columns={
            "chr": "Chromosome",
            "start": "Start",
            "end": "End",
            "clone": "true_clone",
        }
    )

    logger.info(f"Read {root}/simulated_data_related/{sample_id}/{fname}")

    copy_num_columns = truth.columns[3:]

    # NB retain only the true segments that show a CNA for at least one clone.
    if cna_only:
        truth = truth[~(truth[copy_num_columns].eq(1).all(axis=1))]

        logger.warning("Assuming a study of true CNA only.")

    # NB remap normal to clone 0 and increment by 1 otherwise.
    truth = truth.rename(columns=remap_clone_num(copy_num_columns))

    # NB expand to per-spot CNA truth ...
    logger.info(f"Creating table of true CNAs for all spots and segments.")

    expanded_rows = []

    for _, seg in truth.iterrows():
        for spot in spots:
            clone_label = spot_to_clone[spot]
            clone_num = int(clone_label.split("_")[-1])

            expanded_rows.append(
                {
                    "Chromosome": seg["Chromosome"],
                    "Start": seg["Start"],
                    "End": seg["End"],
                    "barcode": spot,
                    "true_clone": clone_num,
                    "true_A": seg.get(f"clone_{clone_num}_A"),
                    "true_B": seg.get(f"clone_{clone_num}_B"),
                }
            )

    spot_truth = pd.DataFrame(expanded_rows)

    # NB some spots will be normal, despite at least one clone having a CNA
    #    here.  Retain only the true segments that show a CNA.
    if cna_only:
        spot_truth = spot_truth[~(spot_truth[["true_A", "true_B"]].eq(1).all(axis=1))]

    spot_truth.insert(3, "sample_id", sample_id)
    spot_truth = pr.PyRanges(spot_truth)

    logger.info(f"Found true CNAs:\n{spot_truth}")

    return spot_truth


def get_sample_loglike(root, sample_id, method, rectangle):
    clone_rectangle = f"clone3_rectangle{rectangle}_w1.0"
    rdr_baf_paths = sorted(
        glob.glob(
            f"{root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/rdrbaf_final_nstates*_smp.npz"
        )
    )

    if len(rdr_baf_paths) == 0:
        logger.warning(
            f"Failed to find rdr_baf path for {root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}"
        )
        return

    rdr_baf = dict(
        np.load(rdr_baf_paths[0]),
        allow_pickle=True,
    )

    return float(rdr_baf["total_llf"])


def get_sample_estimate_pd(root, sample_id, method, rectangle, cna_only=False):
    if rectangle is None:
        parent = f"{root}/nomixing_{method}_related/{sample_id}/"
    else:
        clone_rectangle = f"clone3_rectangle{rectangle}_w1.0"
        parent = f"{root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/"
        
    logger.info(
        f"Solving for clone estimate: {parent}/clone_labels.tsv"
    )

    # NB
    # barcode sample_id       x       y       clone_label
    # spot_0  0       0       0       3
    usecols = ["BARCODES", "clone_label"]

    clones = pd.read_csv(
        f"{parent}/clone_labels.tsv",
        sep="\t",
    ).rename(columns={"clone_label": "clone", "BARCODES": "barcode"})

    spots = clones["barcode"].unique()
    spot_to_clone = dict(zip(clones["barcode"], clones["clone"].astype(int)))

    calls = pd.read_csv(
        f"{parent}/cnv_diploid_seglevel.tsv",
        sep="\t",
    ).rename(columns={"CHR": "Chromosome", "START": "Start", "END": "End"})

    copy_num_columns = calls.columns[3:]

    # NB only CalicoST segments that show CNA for at least one clone.
    if cna_only:
        calls = calls[~(calls[copy_num_columns].eq(1).all(axis=1))]

    # NB clone 0 -> clone_0 etc.
    calls.columns = calls.columns.str.replace(
        r"clone(\d+)\s+([AB])", r"clone_\1_\2", regex=True
    )

    # NB truth per spot, per segment ...
    logger.info(f"Creating table of estimated CNAs for all spots and segments.")

    calls_expanded_rows = []

    for _, seg in calls.iterrows():
        interim = {
            "Chromosome": seg["Chromosome"],
            "Start": seg["Start"],
            "End": seg["End"],
        }

        for spot in spots:
            clone_label = spot_to_clone[spot]
            calls_expanded_rows.append(
                interim
                | {
                    "barcode": spot,
                    "clone": spot_to_clone[spot],
                    "A": seg.get(f"clone_{clone_label}_A"),
                    "B": seg.get(f"clone_{clone_label}_B"),
                }
            )

    spot_cna = pd.DataFrame(calls_expanded_rows)
    spot_cna.insert(3, "sample_id", sample_id)
    spot_cna = pr.PyRanges(spot_cna)

    logger.info(f"Found {method} estimated CNAs:\n{spot_cna}")

    return spot_cna


def get_sample_estimate(root, sample_id, method, rectangle, cna_only=False):
    if rectangle is None:
        parent = f"{root}/nomixing_{method}_related/{sample_id}/"
    else:
        clone_rectangle = f"clone3_rectangle{rectangle}_w1.0"
        parent = f"{root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/"

    logger.info(
        f"Solving for clone estimate: {parent}/clone_labels.tsv"
    )

    clones = (
        pl.read_csv(
            f"{parent}/clone_labels.tsv",
            separator="\t",
        )
        .rename({"clone_label": "clone", "BARCODES": "barcode"}, strict=False)
        .select(["barcode", "clone"])
    )

    logger.info(
        f"Solving for calls: {parent}/cnv_diploid_seglevel.tsv"
    )

    calls = pl.read_csv(
        f"{parent}/cnv_diploid_seglevel.tsv",
        separator="\t",
    ).rename({"CHR": "Chromosome", "START": "Start", "END": "End"})

    copy_num_columns = [
        c for c in calls.columns if c not in ["Chromosome", "Start", "End"]
    ]

    logger.info(f"Found copy num. columns: {copy_num_columns}")

    if cna_only:
        mask = pl.any_horizontal([pl.col(c) != 1 for c in copy_num_columns])
        calls = calls.filter(mask)

    col_rename = {}
    for c in calls.columns:
        if c.startswith("clone"):
            new_name = c.replace("clone", "clone_").replace(" ", "_")
            col_rename[c] = new_name

    if col_rename:
        logger.info(f"Renaming columns: {col_rename}")

        calls = calls.rename(col_rename)

    logger.info(f"Creating table of estimated CNAs for all spots and segments.")

    # NB cross join: all segments × all spots, i.e. replicates each segment for all spots.
    spot_cna = calls.join(clones, how="cross")

    cn_cols = [
        c for c in calls.columns if c.startswith("clone_") and ("_A" in c or "_B" in c)
    ]
    unique_clones = sorted(set(int(c.split("_")[1]) for c in cn_cols))

    logger.info(f"Found unique clones={unique_clones}")

    # NB build when-then expressions to select A/B based on clone, according to given spot, segment.
    if len(unique_clones) > 0:
        a_expr = pl.when(pl.col("clone") == unique_clones[0]).then(
            pl.col(f"clone_{unique_clones[0]}_A")
        )
        b_expr = pl.when(pl.col("clone") == unique_clones[0]).then(
            pl.col(f"clone_{unique_clones[0]}_B")
        )

        for clone_id in unique_clones[1:]:
            a_expr = a_expr.when(pl.col("clone") == clone_id).then(
                pl.col(f"clone_{clone_id}_A")
            )
            b_expr = b_expr.when(pl.col("clone") == clone_id).then(
                pl.col(f"clone_{clone_id}_B")
            )

        a_expr = a_expr.otherwise(None).alias("A")
        b_expr = b_expr.otherwise(None).alias("B")

        spot_cna = spot_cna.with_columns([a_expr, b_expr])
    else:
        # No clone columns found, add null A/B
        spot_cna = spot_cna.with_columns(
            [pl.lit(None).alias("A"), pl.lit(None).alias("B")]
        )

    spot_cna = spot_cna.select(
        ["Chromosome", "Start", "End", "barcode", "clone", "A", "B"]
    )

    spot_cna = spot_cna.with_columns(pl.lit(sample_id).alias("sample_id"))
    spot_cna = spot_cna.select(
        ["Chromosome", "Start", "End", "sample_id", "barcode", "clone", "A", "B"]
    )

    # TODO convert to pandas for PyRanges compatibility
    spot_cna = pr.PyRanges(spot_cna.to_pandas())

    logger.info(f"Found {method} estimated CNAs:\n{spot_cna}")

    return spot_cna


def get_best_sample_estimate(root, sample_id, method, cna_only=False):
    best_rectangle, best_loglike = None, -np.inf

    for rectangle in range(10):
        loglike = get_sample_loglike(root, sample_id, method, rectangle)

        if loglike is None:
            if rectangle < 5:
                logger.warning("Expected at least 5 initializations")
            break

        if loglike >= best_loglike:
            best_rectangle = rectangle
            best_loglike = loglike

    logger.info(
        f"Found best {method} initialization={best_rectangle} with loglike={best_loglike:.6e}"
    )

    best_spot_cna = get_sample_estimate(
        root,
        sample_id,
        method,
        best_rectangle,
        cna_only=cna_only,
    )

    return best_rectangle, best_spot_cna, best_loglike


def get_join(first, second):
    # NB left-join overlaps of second on first, e.g. estimated CNA intervals on
    #    the known, truth intervals.  Sample_id refers to different simulated realizations/
    #    true CNA configuration.
    result = first.join_overlaps(
        second,
        match_by=["barcode", "sample_id"],
        join_type="left",
        multiple="all",
        report_overlap_column="overlap_bp",
        slack=0,
    )

    start_b = result.pop("Start_b")
    end_b = result.pop("End_b")

    result.insert(3, "Start_b", start_b)
    result.insert(4, "End_b", end_b)

    # NB we will live with NANs on join eventually, so float.
    for col in ["true_clone", "true_A", "true_B"]:
        result[col] = result[col].astype(float)

    # NB was there a successful join?
    invalid = ~np.isfinite(result["A"])

    result.loc[invalid, "overlap_bp"] = 0.0
    result["overlap_frac"] = result["overlap_bp"] / (result["End"] - result["Start"])

    logger.info(f"Found joint CNAs:\n{result}")

    return result


def compute_state_oversampling_rate(tsv_path):
    """
    Compute state assignment counts: for each unique integer (A,B) state,
    find the maximum number of times it's assigned across all clones.
    
    Args:
        tsv_path: Path to cnv_diploid_perstate.tsv file
        
    Returns:
        Tuple of (num_clones, state_max_counts) where:
        - num_clones: number of clones in the data
        - state_max_counts: Dict mapping state tuples like (1,1), (2,1), etc. 
          to the maximum count of that state across all clones
    """
    df = pd.read_csv(tsv_path, sep="\t")
    
    # Identify clone columns
    clone_cols = [c for c in df.columns if " A" in c or " B" in c]
    
    # Extract unique clone prefixes
    clones = sorted(set(c.rsplit(" ", 1)[0] for c in clone_cols))
    num_clones = len(clones)
    
    # Collect all unique states across all clones
    all_states = set()
    for clone in clones:
        a_col = f"{clone} A"
        b_col = f"{clone} B"
        if a_col in df.columns and b_col in df.columns:
            states = list(zip(df[a_col], df[b_col]))
            all_states.update(states)
    
    # Calculate max count for each state across all clones
    state_max_counts = {}
    for state in sorted(all_states):
        counts = []
        for clone in clones:
            a_col = f"{clone} A"
            b_col = f"{clone} B"
            if a_col in df.columns and b_col in df.columns:
                is_state = (df[a_col] == state[0]) & (df[b_col] == state[1])
                count = int(is_state.sum())
                counts.append(count)
        state_max_counts[state] = int(max(counts)) if counts else 0
    
    return num_clones, state_max_counts


def get_validation_stats(
    sample_id, best_rectangle, best_loglike, spot_join_cna, include_flip=True, num_clones=None, state_oversampling_rates=None
):
    # NB was there an interval called on this truth segment?
    matched = np.isfinite(spot_join_cna["A"])
    match_rate = matched.mean()
    
    # NB limit to the matches only.                                                                                                                                                                                                                    
    match_spot_join_cna = spot_join_cna[matched]
    num_match = len(match_spot_join_cna)
    
    # NB did we recover the true CNA (up to a phase flip)?
    correct_match = (match_spot_join_cna["A"] == match_spot_join_cna["true_A"]) & (
        match_spot_join_cna["B"] == match_spot_join_cna["true_B"]
    )
    correct_rate = correct_match.mean()

    if include_flip:
        correct_match = correct_match | (
            match_spot_join_cna["B"] == match_spot_join_cna["true_A"]
        ) & (match_spot_join_cna["A"] == match_spot_join_cna["true_B"])

    is_normal = (match_spot_join_cna["true_A"] == 1) & (match_spot_join_cna["true_B"] == 1)

    normal_recovery = correct_match & is_normal
    normal_recovery_rate = np.count_nonzero(normal_recovery) / np.count_nonzero(is_normal)
    
    cna_recovery = correct_match & ~is_normal
    cna_recovery_rate = np.count_nonzero(cna_recovery) / np.count_nonzero(~is_normal)
        
    cna_false_positive = ~correct_match & is_normal
    cna_false_positive_rate = np.count_nonzero(cna_false_positive) / np.count_nonzero(is_normal)
    
    # TODO
    try:
        ari = adjusted_rand_score(
            spot_join_cna["true_clone"].astype(int), 
            spot_join_cna["clone"].astype(int)
        )
    except:
        ari = np.nan
        
    best_clone_mapping, clone_mapping_success_rate, _ = best_permutation_accuracy(
        spot_join_cna["true_clone"].to_numpy(), spot_join_cna["clone"].to_numpy()
    )

    logger.info(
        f"Found normal rate={is_normal.mean():.3f}, match rate={match_rate:.3f}, correct_rate={correct_rate}, ari={ari:.6f}, mapped fraction={clone_mapping_success_rate:.4f}, normal recovery rate={normal_recovery_rate:.3f}, cna recovery rate={cna_recovery_rate:.3f} and cna false positive rate={cna_false_positive_rate:.3f} for include_flip={include_flip}."
    )
    
    logger.info(f"Found best clone 1-1 mapping: {best_clone_mapping}")

    # NB distribution of true clone in matches, required for normalization.
    clone_marginals = dict(
        sorted(Counter(match_spot_join_cna["true_clone"].astype(int)).items())
    )
    clone_transitions = dict(
        sorted(
            Counter(
                zip(
                    match_spot_join_cna["true_clone"].astype(int),
                    match_spot_join_cna["clone"].astype(int),
                )
            ).items()
        )
    )

    logger.info(f"Found clone marginals:\n{clone_marginals}")
    logger.info(f"Found clone transition rates:")

    # NB normalized to answer the question: what happened to a given true clone?
    for (true_clone, pred_clone), count in sorted(clone_transitions.items()):
        frac = (
            count / clone_marginals[int(true_clone)]
            if clone_marginals[int(true_clone)] > 0
            else np.nan
        )
        logger.info(
            f"\t{true_clone}->{pred_clone}\t{count}\t{count / num_match:.4f}\t{frac:<10.4f}"
        )

    # NB normalized to answer the question: what happened to a given true CNA?
    cna_marginals = dict(
        sorted(
            Counter(
                zip(
                    match_spot_join_cna["true_A"].astype(int),
                    match_spot_join_cna["true_B"].astype(int),
                )
            ).items()
        )
    )
    cna_marginal_rates = {
        f"({k[0]},{k[1]})": v / num_match 
        for k, v in cna_marginals.items()
    }
    
    cna_transitions = dict(
        sorted(
            Counter(
                zip(
                    match_spot_join_cna["true_A"].astype(int),
                    match_spot_join_cna["true_B"].astype(int),
                    match_spot_join_cna["A"].astype(int),
                    match_spot_join_cna["B"].astype(int),
                )
            ).items()
        )
    )

    logger.info(f"Found cna marginals:\n{cna_marginals}")
    logger.info(f"Found cna transition rates:")

    for (true_a, true_b, pred_a, pred_b), count in sorted(cna_transitions.items()):
        frac = (
            count / cna_marginals[(true_a, true_b)]
            if cna_marginals[(true_a, true_b)] > 0
            else np.nan
        )

        true_pair = f"({true_a},{true_b})"
        pred_pair = f"({pred_a},{pred_b})"

        logger.info(
            f"\t{true_pair}->{pred_pair}\t{count}\t{count / num_match:.6f}\t{frac:.6f}"
        )

    # Convert state_oversampling_rates keys to strings for YAML serialization
    state_oversampling_dict = {}
    if state_oversampling_rates:
        state_oversampling_dict = {f"({k[0]},{k[1]})": int(v) for k, v in state_oversampling_rates.items()}

    return {
        "sample_id": sample_id,
        "initialization": best_rectangle,
        "loglike": best_loglike,
        "num_clones": int(num_clones) if num_clones is not None else None,
        "state_max_counts": state_oversampling_dict,
        "normal_rate": float(is_normal.mean()),
        "match_rate": float(match_rate),
        "correct_rate": float(correct_rate),
        "ari": float(ari),
        "clone_mapping_success_rate": float(clone_mapping_success_rate),
        "normal_recovery_rate": float(normal_recovery_rate),
        "cna_recovery_rate": float(cna_recovery_rate),
        "cna_false_positive_rate": float(cna_false_positive_rate),
        "include_flip": include_flip,
        "best_clone_mapping": {int(k): int(v) for k, v in best_clone_mapping.items()},
        "clone_marginals": dict(clone_marginals),
        "clone_transitions": dict(
            {f"{k[0]}->{k[1]}": v for k, v in clone_transitions.items()}
        ),
        "clone_transition_rates": dict(
            {f"{k[0]}->{k[1]}": v/clone_marginals[k[0]] for k, v in clone_transitions.items()}
        ),
        "cna_marginals": dict(
            {f"({k[0]},{k[1]})": v for k, v in cna_marginals.items()}
        ),
        "cna_marginal_rates": cna_marginal_rates,
        "cna_transitions": dict(
            {f"({k[0]},{k[1]})->({k[2]},{k[3]})": v for k, v in cna_transitions.items()}
        ),
        "cna_transition_rates": dict(
            {f"({k[0]},{k[1]})->({k[2]},{k[3]})": v/cna_marginals[(k[0],k[1])] for k, v in cna_transitions.items()}
        ),
    }


def save_validation_stats_yaml(stats, output_path):
    output_path = Path(output_path)

    logger.info("Saving validation stats to %s:\n%s", output_path, pformat(stats, width=100))

    with open(output_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)


def main():
    # root = "/u/mw9568/scratch/calicost_sims"
    root = "/Users/mw9568/Work/ragr/sim"
    use_cache = False

    gene_ranges = read_gene_ranges()

    method = "cnaster"
    # method = "calicost"

    # "numcnas1.2_cnasize1e7_ploidy2_random0",
    # "numcnas3.3_cnasize3e7_ploidy2_random0",
    # "numcnas3.3_cnasize5e7_ploidy2_random0",

    sample_ids = [
         "numcnas3.3_cnasize3e7_ploidy2_random0",
    ]
    
    # sample_ids = [xx.split("/")[-1] for xx in sorted(glob.glob(f"{root}/nomixing_{method}_related/*"))]
                  
    logger.info(
        f"Analyzing with {method} the sample_ids={sample_ids} simulations @\n{root}"
    )
                  
    for sample_id in sample_ids:
        if use_cache and Path(f"{root}/stats/{method}/validation_stats_{sample_id}.yaml").exists():
            logger.warning(f"Utilizing existing validation stats for {sample_id}.")
            continue

        try:
            spot_truth_cna = get_sample_truth(root, sample_id)
            gene_spot_truth_cna = spot_truth_cna.overlap(gene_ranges)

            logger.info(
                f"Found overlap rate of truth CNAs with genes={len(gene_spot_truth_cna) / len(spot_truth_cna):.3f}"
            )
            
            best_rectangle, spot_calicost_cna, best_loglike = get_best_sample_estimate(
                root,
                sample_id,
                method,
            )
            
            if best_rectangle is not None:
                clone_rectangle = f"clone3_rectangle{best_rectangle}_w1.0"
                tsv_path = Path(root) / f"nomixing_{method}_related" / sample_id / clone_rectangle / f"cnv_diploid_perstate.tsv"
            else:
                tsv_path = Path(root) / f"nomixing_{method}_related" / sample_id / f"cnv_diploid_perstate.tsv"
                
            if tsv_path.exists():
                num_clones, state_max_counts = compute_state_oversampling_rate(tsv_path)
                logger.info(f"Computed num_clones={num_clones}, state_max_counts={state_max_counts}")
            else:
                num_clones = None
                state_max_counts = {}
                logger.warning(f"Could not find cnv_diploid_perstate.tsv at {tsv_path}")

            spot_truth_cna_match = get_join(spot_truth_cna, spot_calicost_cna)

            validation_stats = get_validation_stats(
                sample_id, best_rectangle, best_loglike, spot_truth_cna_match, 
                num_clones=num_clones,
                state_oversampling_rates=state_max_counts
            )
            
            save_validation_stats_yaml(
                validation_stats, f"{root}/stats/{method}/validation_stats_{sample_id}.yaml"
            )
        except Exception as E:
            logger.warning(f"Failed on {sample_id} with error:\n{E}")
                        
    logger.info("\n\nDone.\n")


if __name__ == "__main__":
    main()
