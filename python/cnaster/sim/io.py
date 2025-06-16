import ast
import gzip
import pandas as pd

def get_truth(config, run_id, sample_name):
    tpath = f"{config.output_dir}/run{run_id}/{sample_name}/truth/{sample_name}.tsv.gz"

    with gzip.open(tpath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                names = ast.literal_eval(line[1:].strip())
                break

    return  pd.read_csv(tpath, sep="\t", comment="#",  names=names)