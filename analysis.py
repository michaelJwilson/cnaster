import pandas as pd
import pyranges as pr

def remap_columns(columns):
    new_columns = {}
    
    for col in columns:
        if col.startswith('normal_'):
            # normal -> true_clone_0
            new_col = col.replace('normal_', 'true_clone_0_').replace('_copy', '')
            new_columns[col] = new_col
        elif col.startswith('clone_'):
            # Extract clone number and increment by 1
            parts = col.split('_')
            clone_num = int(parts[1]) + 1
            allele = parts[2]  # A or B
            new_columns[col] = f'tru_clone_{clone_num}_{allele}'

    return new_columns
            
# NB e.g. {truth_cna.tsv}
fname = "truth_acn_profile.tsv"

#         labels  x       y
# spot_0  clone_2 0       0
truth_clones = pd.read_csv("~/scratch/calicost_sims/simulated_data_related/numcnas1.2_cnasize1e7_ploidy2_random0/truth_clone_labels.tsv", sep="\t")

print(truth_clones)

# clone	        chr	start	        end	        A_copy	B_copy
# clone_0	20	51816053	61816053	0	1
truth = pd.read_csv(
    f"~/scratch/calicost_sims/simulated_data_related/numcnas1.2_cnasize1e7_ploidy2_random0/{fname}",
    sep="\t",
).rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})
copy_num_columns = truth.columns[3:]

truth[copy_num_columns] = truth[copy_num_columns].astype('int8')

# NB entire rest of genome is the normal state.
truth_cna = truth[~(truth[copy_num_columns].eq(1).all(axis=1))]
truth_cna = truth_cna.rename(columns=remap_columns(copy_num_columns))
truth_cna = pr.PyRanges(truth_cna)

print(truth_cna)

exit(0)

# barcode sample_id       x       y       clone_label
# spot_0  0       0       0       3

calicost_clones = pd.read_csv("~/scratch/calicost_sims/nomixing_calicost_related/numcnas1.2_cnasize1e7_ploidy2_random0/clone_labels.tsv")

calicost = pd.read_csv(
    "~/scratch/calicost_sims/nomixing_calicost_related/numcnas1.2_cnasize1e7_ploidy2_random0/cnv_seglevel.tsv",
    sep="\t",
).rename(columns={"CHR": "Chromosome", "START": "Start", "END": "End"})

copy_num_columns = calicost.columns[3:]

calicost_cna = calicost[~(calicost[copy_num_columns].eq(1).all(axis=1))]

calicost_cna.columns = calicost_cna.columns.str.replace(r'clone(\d+)\s+([AB])', r'clone_\1_\2', regex=True)
calicost_cna = pr.PyRanges(calicost_cna)

join_cna = truth_cna.join_overlaps(calicost_cna)

start_b = join_cna.pop('Start_b')
end_b = join_cna.pop('End_b')

join_cna.insert(3, 'Start_b', start_b)
join_cna.insert(4, 'End_b', end_b)

print(join_cna)
