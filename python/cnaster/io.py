import ast
import gzip
import pandas as pd

def get_spots(config, run_id, sample_name):
    mpath = f"{config.output_dir}/run{run_id}/{sample_name}/{sample_name}_visium.tsv.gz"

    with gzip.open(mpath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                names = line[1:].strip().split()
                break

    return  pd.read_csv(mpath, sep="\t", comment="#",  names=names)