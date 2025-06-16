import ast
import gzip
import pandas as pd

def get_meta(config, run_id, sample_name):
    mpath = f"{config.output_dir}/run{run_id}/{sample_name}/meta/{sample_name}.tsv.gz"

    with gzip.open(mpath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                names = ast.literal_eval(line[1:].strip())
                break

    return  pd.read_csv(mpath, sep="\t", comment="#",  names=names)