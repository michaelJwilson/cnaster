import time
import logging
import numpy as np
import pandas as pd
import pyranges as pr
from collections import Counter
from sklearn.metrics import adjusted_rand_score

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


def get_sample_estimate(
    root, sample_id, rectangle=0, method="calicost", cna_only=False
):
    clone_rectangle = f"clone3_rectangle{rectangle}_w1.0"

    logger.info(
        f"Solving for clone estimate: {root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/clone_labels.tsv"
    )

    # NB
    # barcode sample_id       x       y       clone_label
    # spot_0  0       0       0       3
    usecols = ["BARCODES", "clone_label"]

    clones = pd.read_csv(
        f"{root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/clone_labels.tsv",
        sep="\t",
    ).rename(columns={"clone_label": "clone", "BARCODES": "barcode"})

    spots = clones["barcode"].unique()
    spot_to_clone = dict(zip(clones["barcode"], clones["clone"].astype(int)))

    calls = pd.read_csv(
        f"{root}/nomixing_{method}_related/{sample_id}/{clone_rectangle}/cnv_seglevel.tsv",
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


def get_join(first, second):
    # NB left-join overlaps of second on first, e.g. estimated CNA intervals on
    #    the known, truth intervals.  Sample_id refers to different simulated realizations/
    #    true CNA configuration.
    result = first.join_overlaps(
        second,
        match_by=["barcode", "sample_id"],
        join_type="left",
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


def get_success_rate(spot_join_cna, include_flip=True):
    match = np.isfinite(spot_join_cna["A"])

    # NB did we recover the true CNA (up to a phase flip)?
    correct_match = (spot_join_cna["A"] == spot_join_cna["true_A"]) & (
        spot_join_cna["B"] == spot_join_cna["true_B"]
    )

    if include_flip:
        correct_match = correct_match | (
            spot_join_cna["B"] == spot_join_cna["true_A"]
        ) & (spot_join_cna["A"] == spot_join_cna["true_B"])

    is_normal = (spot_join_cna["true_A"] == 1) & (spot_join_cna["true_B"] == 1)

    normal_recovery = correct_match & is_normal

    cna_recovery = correct_match & ~is_normal
    cna_false_positive = ~correct_match & is_normal

    # NB was there an interval called on this truth segment?
    match_rate = match.mean()

    ari = adjusted_rand_score(spot_join_cna["true_clone"], spot_join_cna["clone"])

    logger.info(
        f"Found normal rate={is_normal.mean():.3f}, match rate={match_rate:.3f} with ari={ari:.6f}, normal recovery rate={normal_recovery.mean():.3f}, cna recovery rate={cna_recovery.mean():.3f} and cna false positive rate={cna_false_positive.mean():.3f} for include_flip={include_flip}."
    )

    # NB limit to the matches only.
    match_spot_join_cna = spot_join_cna[match]
    num_match = len(match_spot_join_cna)

    # NB distribution of true clone in matches, required for normalization.
    clone_marginals = Counter(match_spot_join_cna["true_clone"].astype(int))
    clone_transitions = Counter(
        zip(
            match_spot_join_cna["true_clone"].astype(int),
            match_spot_join_cna["clone"].astype(int),
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
    cna_marginals = Counter(
        zip(
            match_spot_join_cna["true_A"].astype(int),
            match_spot_join_cna["true_B"].astype(int),
        )
    )
    cna_transitions = Counter(
        zip(
            match_spot_join_cna["true_A"].astype(int),
            match_spot_join_cna["true_B"].astype(int),
            match_spot_join_cna["A"].astype(int),
            match_spot_join_cna["B"].astype(int),
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
            f"\t{true_pair}->{pred_pair}\t{count}\t{count / num_match:.4f}\t{frac:.6e}"
        )

    return


def main():
    root = "~/scratch/calicost_sims/"

    # method = "cnaster"
    method = "calicost"
    rectangle = 0

    # "numcnas1.2_cnasize1e7_ploidy2_random0",
    # "numcnas3.3_cnasize3e7_ploidy2_random0",
    # "numcnas3.3_cnasize5e7_ploidy2_random0",

    sample_ids = [
        "numcnas1.2_cnasize1e7_ploidy2_random0",
    ]

    logger.info(
        f"Analyzing with {method} the sample_ids={sample_ids} (for rectangle={rectangle}) simulations @\n{root}"
    )

    result = []

    for sample_id in sample_ids[:1]:
        spot_truth_cna = get_sample_truth(root, sample_id)
        spot_calicost_cna = get_sample_estimate(
            root, sample_id, method=method, rectangle=rectangle
        )

        spot_truth_cna_match = get_join(spot_truth_cna, spot_calicost_cna)
        # spot_calicost_cna_match = get_join(spot_calicost_cna, spot_truth_cna)

        result.append(spot_truth_cna_match)

    result = pr.concat(result)
    success_rate = get_success_rate(result)

    logger.info("\n\nDone.\n")

    
if __name__ == "__main__":
    main()
