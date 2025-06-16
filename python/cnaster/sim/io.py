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

def get_exp_baseline(config):
    epath = f"{config.output_dir}/baseline/expression.tsv.gz"
    return pd.read_csv(epath, sep="\t", comment="#", names=None).to_numpy()

def get_snp_baseline(config):
    spath = f"{config.output_dir}/baseline/snps.tsv.gz"
    return pd.read_csv(spath, sep="\t", comment="#", names=None).to_numpy()