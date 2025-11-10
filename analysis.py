import logging
import numpy as np
import pandas as pd
import pyranges as pr

logger = logging.getLogger(__name__)


def remap_clone_num(entries):
    entries = np.unique(entries)

    logger.info(f"Found unique entries: {entries}")

    has_normal = np.any(["normal" in xx for xx in entries])
    clone_zp = 1 if has_normal else 0

    new_entries = {}

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


def get_sample_truth(root, sample_id):
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

    copy_num_columns = truth.columns[3:]

    # NB retain only the true segments that show a CNA for at least one clone.
    truth_cna = truth[~(truth[copy_num_columns].eq(1).all(axis=1))]

    # NB remap normal to clone 0 and increment by 1 otherwise.
    truth_cna = truth_cna.rename(columns=remap_clone_num(copy_num_columns))

    # NB expand to per-spot CNA truth ...
    expanded_rows = []

    for _, seg in truth_cna.iterrows():
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

    spot_truth_cna = pd.DataFrame(expanded_rows)

    # NB retain only the true segments that show a CNA.
    spot_truth_cna = spot_truth_cna[
        ~(spot_truth_cna[["true_A", "true_B"]].eq(1).all(axis=1))
    ]

    spot_truth_cna.insert(3, "sample_id", sample_id)
    spot_truth_cna = pr.PyRanges(spot_truth_cna)

    logger.info(f"Found true CNAs:\n{spot_truth_cna}")

    return spot_truth_cna


def get_sample_calicost(root, sample_id):
    # TODO !!
    clone_rectangle = "clone3_rectangle0_w1.0"

    logger.info(
        f"Solving for {root}/nomixing_calicost_related/{sample_id}/{clone_rectangle}/clone_labels.tsv"
    )

    # NB
    # barcode sample_id       x       y       clone_label
    # spot_0  0       0       0       3
    usecols = ["BARCODES", "clone_label"]

    calicost_clones = pd.read_csv(
        f"{root}/nomixing_calicost_related/{sample_id}/{clone_rectangle}/clone_labels.tsv",
        sep="\t",
        usecols=usecols,
    ).rename(columns={"clone_label": "clone", "BARCODES": "barcode"})

    spots = calicost_clones["barcode"].unique()
    spot_to_calicost_clone = dict(
        zip(calicost_clones["barcode"], calicost_clones["clone"])
    )

    # logger.info(
    # f"Found {calicost_clones['clone'].unique()} clones in CalicoST for sample {sample_id}."
    # )

    calicost = pd.read_csv(
        f"{root}/nomixing_calicost_related/{sample_id}/{clone_rectangle}/cnv_seglevel.tsv",
        sep="\t",
    ).rename(columns={"CHR": "Chromosome", "START": "Start", "END": "End"})

    copy_num_columns = calicost.columns[3:]

    # NB only CalicoST segments that show CNA for at least one clone.
    calicost_cna = calicost[~(calicost[copy_num_columns].eq(1).all(axis=1))]

    # NB clone 0 -> clone_0 etc.
    calicost_cna.columns = calicost_cna.columns.str.replace(
        r"clone(\d+)\s+([AB])", r"clone_\1_\2", regex=True
    )

    # NB truth per spot, per segment ...
    calicost_expanded_rows = []

    for _, seg in calicost_cna.iterrows():
        for spot in spots:
            clone_label = spot_to_calicost_clone[spot]
            calicost_expanded_rows.append(
                {
                    "Chromosome": seg["Chromosome"],
                    "Start": seg["Start"],
                    "End": seg["End"],
                    "barcode": spot,
                    "clone": int(spot_to_calicost_clone[spot]),
                    "A": seg.get(f"clone_{clone_label}_A"),
                    "B": seg.get(f"clone_{clone_label}_B"),
                }
            )

    spot_calicost_cna = pd.DataFrame(calicost_expanded_rows)
    spot_calicost_cna.insert(3, "sample_id", sample_id)
    spot_calicost_cna = pr.PyRanges(spot_calicost_cna)

    logger.info(f"Found calicost CNAs:\n{spot_calicost_cna}")

    return spot_calicost_cna


def get_join(first, second):
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

    for col in ["true_clone", "true_A", "true_B"]:
        result[col] = result[col].astype(float)

    invalid = ~np.isfinite(result["A"])

    result.loc[invalid, "overlap_bp"] = 0.0
    result["overlap_frac"] = result["overlap_bp"] / (result["End"] - result["Start"])

    logger.info(f"Found joint CNAs:\n{result}")

    return result


def get_success_rate(spot_join_cna, include_flip=True):
    match = np.isfinite(spot_join_cna["A"])

    correct_match = (spot_join_cna["A"] == spot_join_cna["true_A"]) & (
        spot_join_cna["B"] == spot_join_cna["true_B"]
    )

    if include_flip:
        correct_match = correct_match | (
            spot_join_cna["B"] == spot_join_cna["true_A"]
        ) & (spot_join_cna["A"] == spot_join_cna["true_B"])

    match_rate = match.mean()
    success_rate = correct_match.mean()

    logger.info(
        f"Found match rate={match_rate:.3f} with success rate={success_rate:.3f} for include_flip={include_flip}."
    )

    return success_rate


if __name__ == "__main__":
    root = "~/scratch/calicost_sims/"

    sample_ids = [
        "numcnas1.2_cnasize1e7_ploidy2_random0",
        "numcnas3.3_cnasize3e7_ploidy2_random0",
    ]
    result = []

    for sample_id in sample_ids[:1]:
        spot_truth_cna = get_sample_truth(root, sample_id)
        spot_calicost_cna = get_sample_calicost(root, sample_id)

        spot_truth_cna_match = get_join(spot_truth_cna, spot_calicost_cna)
        # spot_calicost_cna_match = get_join(spot_truth_cna, spot_calicost_cna)

        result.append(spot_truth_cna_match)

    result = pr.concat(result)
    success_rate = get_success_rate(result)

    logger.info("\n\nDone.\n\n")
