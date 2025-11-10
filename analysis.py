import logging
import pandas as pd
import pyranges as pr

logger = logging.getLogger(__name__)

def remap_clone_num(columns):
    new_columns = {}

    for col in columns:
        if col.startswith("normal_"):
            # normal -> true_clone_0
            new_col = col.replace("normal_", "true_clone_0_").replace("_copy", "")
            new_columns[col] = new_col
        elif col.startswith("clone_"):
            # Extract clone number and increment by 1
            parts = col.split("_")
            clone_num = int(parts[1]) + 1
            allele = parts[2]  # A or B
            new_columns[col] = f"tru_clone_{clone_num}_{allele}"

    return new_columns

def get_sample_truth(root, sample_id, fname="truth_acn_profile.tsv"):
    # NB 
    #         labels  x       y
    # spot_0  clone_2 0       0
    truth_clones = pd.read_csv(
        f"{root}/simulated_data_related/{sample_id}/truth_clone_labels.tsv",
        sep="\t",
        names=["barcode", "label", "x", "y"],
        skiprows=1,
    )

    truth_clones["label"] = truth_clones["label"].str.replace("clone", "true_clone")

    spots = truth_clones["barcode"].unique()
    spot_to_clone = dict(zip(truth_clones["barcode"], truth_clones["label"]))

    logger.info(f"Found {truth_clones['label'].unique()} clones in truth for sample {sample_id} spaceranger.")

    # NB
    # clone	    chr	    start	    end	        A_copy	B_copy
    # clone_0	20	    51816053	61816053	0	    1
    truth = pd.read_csv(
        f"{root}/simulated_data_related/{sample_id}/{fname}",
        sep="\t",
    ).rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})

    # NB retain only the true segments that show a CNA for at least one clone.
    copy_num_columns = truth.columns[3:]

    truth_cna = truth[~(truth[copy_num_columns].eq(1).all(axis=1))]

    # NB remap normal to clone 0 and increment by 1 otherwise.
    truth_cna = truth_cna.rename(columns=remap_clone_num(copy_num_columns))

    logger.info(f"Found {truth_cna['label'].unique()} clones in truth for sample {sample_id} cna.")

    assert truth_clones['label'].unique() == truth_cna['label'].unique(), "Mismatch between clone labels in truth clones and truth cna: {truth_clones['label'].unique()} != {truth_cna['label'].unique()}"

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
                    "true_A": seg.get(f"true_clone_{clone_num}_A"),
                    "true_B": seg.get(f"true_clone_{clone_num}_B"),
                }
            )

    spot_truth_cna = pd.DataFrame(expanded_rows)
    spot_truth_cna = pr.PyRanges(spot_truth_cna)
    spot_truth_cna.insert(3, "sample_id", sample_id)

    return spot_truth_cna

def get_sample_calicost(root, sample_id):
    # NB 
    # barcode sample_id       x       y       clone_label
    # spot_0  0       0       0       3
    calicost_clones = pd.read_csv(
        f"{root}/nomixing_calicost_related/{sample_id}/clone_labels.tsv",
        sep="\t",
        usecols=["barcode", "x", "y", "clone_label"],
    ).rename(columns={"clone_label": "label"})

    spots = calicost_clones["barcode"].unique()
    spot_to_calicost_clone = dict(
        zip(calicost_clones["barcode"], calicost_clones["label"])
    )

    logger.info(f"Found {calicost_clones['label'].unique()} clones in CalicoST for sample {sample_id}.")

    calicost = pd.read_csv(
        f"{root}/nomixing_calicost_related/{sample_id}/cnv_seglevel.tsv",
        sep="\t",
    ).rename(columns={"CHR": "Chromosome", "START": "Start", "END": "End"})

    copy_num_columns = calicost.columns[3:]

    # NB only CalicoST segments that show CNA for at least one clone.
    calicost_cna = calicost[~(calicost[copy_num_columns].eq(1).all(axis=1))]

    # NB clone 0 -> clone_0 etc.
    calicost_cna.columns = calicost_cna.columns.str.replace(
        r"clone(\d+)\s+([AB])", r"clone_\1_\2", regex=True
    )

    assert calicost_clones['label'].unique() == calicost_cna['label'].unique(), "Mismatch between clone labels in calicost clones and calicost cna: {calicost_clones['label'].unique()} != {calicost_cna['label'].unique()}"

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
    spot_calicost_cna = pr.PyRanges(spot_calicost_cna)
    spot_calicost_cna.insert(3, "sample_id", sample_id)

    return spot_calicost_cna

def get_truth_calicost_join(spot_truth_cna, spot_calicost_cna):
    spot_join_cna = spot_truth_cna.join_overlaps(spot_calicost_cna, match_by=["barcode", "sample_id"])

    start_b = spot_join_cna.pop("Start_b")
    end_b = spot_join_cna.pop("End_b")

    spot_join_cna.insert(3, "Start_b", start_b)
    spot_join_cna.insert(4, "End_b", end_b)

    return spot_join_cna


if __name__ == "__main__":
    root = "~/scratch/calicost_sims/"
    sample_id = "numcnas1.2_cnasize1e7_ploidy2_random0"

    spot_truth_cna = get_sample_truth(root, sample_id)
    spot_calicost_cna = get_sample_calicost(root, sample_id)

    spot_join_cna = get_truth_calicost_join(spot_truth_cna, spot_calicost_cna)

    success_rate = (
        (spot_join_cna["A"] == spot_join_cna["true_A"])
        & (spot_join_cna["B"] == spot_join_cna["true_B"])
    ).mean()

    print(success_rate)

    spot_join_cna = spot_join_cna[~(spot_join_cna[["true_A", "true_B"]].eq(1).all(axis=1))]

    print(spot_join_cna)

    success_rate = (
        (spot_join_cna["A"] == spot_join_cna["true_A"])
        & (spot_join_cna["B"] == spot_join_cna["true_B"])
    ).mean()

    print(success_rate)

    logger.info("\n\nDone.\n\n")