import pandas as pd
import pyranges as pr


def remap_columns(columns):
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


# NB e.g. {truth_cna.tsv}
fname = "truth_acn_profile.tsv"

#         labels  x       y
# spot_0  clone_2 0       0
truth_clones = pd.read_csv(
    "~/scratch/calicost_sims/simulated_data_related/numcnas1.2_cnasize1e7_ploidy2_random0/truth_clone_labels.tsv",
    sep="\t",
    names=["barcode", "label", "x", "y"],
    skiprows=1,
)
truth_clones["label"] = truth_clones["label"].str.replace("clone", "tru_clone")

# clone	        chr	start	        end	        A_copy	B_copy
# clone_0	20	51816053	61816053	0	1
truth = pd.read_csv(
    f"~/scratch/calicost_sims/simulated_data_related/numcnas1.2_cnasize1e7_ploidy2_random0/{fname}",
    sep="\t",
).rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})
copy_num_columns = truth.columns[3:]

truth[copy_num_columns] = truth[copy_num_columns].astype("int8")

# NB entire rest of genome is the normal state.
truth_cna = truth[~(truth[copy_num_columns].eq(1).all(axis=1))]
truth_cna = truth_cna.rename(columns=remap_columns(copy_num_columns))
truth_cna = pr.PyRanges(truth_cna)

# print(truth_cna)

spot_to_clone = dict(zip(truth_clones["barcode"], truth_clones["label"]))
spots = truth_clones["barcode"].unique()

# Expand: create one row per (segment, spot) combination
expanded_rows = []

for _, seg in truth_cna.iterrows():
    for spot in spots:
        # Get clone assignment for this spot
        clone_label = spot_to_clone[spot]

        # Determine which A/B columns to use based on clone
        if clone_label == "normal":
            a_col = "tru_clone_0_A"
            b_col = "tru_clone_0_B"
        else:
            # Extract clone number (e.g., "tru_clone_2" -> 2)
            clone_num = int(clone_label.split("_")[2])
            # Column names after remap: tru_clone_{N+1}_A
            remapped_num = clone_num + 1
            a_col = f"tru_clone_{remapped_num}_A"
            b_col = f"tru_clone_{remapped_num}_B"

        # Extract A/B copies for this clone (handle missing columns)
        a_copy = seg.get(a_col, 1)  # Default to 1 if column missing
        b_copy = seg.get(b_col, 1)

        expanded_rows.append(
            {
                "Chromosome": seg["Chromosome"],
                "Start": seg["Start"],
                "End": seg["End"],
                "barcode": spot,
                "true_clone": clone_label.split("_")[-1],
                "A": a_copy,
                "B": b_copy,
            }
        )

spot_truth_cna = pd.DataFrame(expanded_rows)
spot_truth_cna = pr.PyRanges(spot_truth_cna)

# print(spot_truth_cna)

# barcode sample_id       x       y       clone_label
# spot_0  0       0       0       3

calicost_clones = pd.read_csv(
    "~/scratch/calicost_sims/nomixing_calicost_related/numcnas1.2_cnasize1e7_ploidy2_random0/clone_labels.tsv",
    sep="\t",
)

print(calicost_clones)

exit(0)

calicost = pd.read_csv(
    "~/scratch/calicost_sims/nomixing_calicost_related/numcnas1.2_cnasize1e7_ploidy2_random0/cnv_seglevel.tsv",
    sep="\t",
).rename(columns={"CHR": "Chromosome", "START": "Start", "END": "End"})

copy_num_columns = calicost.columns[3:]

calicost_cna = calicost[~(calicost[copy_num_columns].eq(1).all(axis=1))]

calicost_cna.columns = calicost_cna.columns.str.replace(
    r"clone(\d+)\s+([AB])", r"clone_\1_\2", regex=True
)
calicost_cna = pr.PyRanges(calicost_cna)

join_cna = truth_cna.join_overlaps(calicost_cna)

start_b = join_cna.pop("Start_b")
end_b = join_cna.pop("End_b")

join_cna.insert(3, "Start_b", start_b)
join_cna.insert(4, "End_b", end_b)

# print(join_cna)
