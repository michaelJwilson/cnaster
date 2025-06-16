import ast
import gzip
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_spots(config, run_id, sample_name):
    mpath = f"{config.output_dir}/run{run_id}/{sample_name}/{sample_name}_visium.tsv.gz"

    logger.info(f"Reading spots from {mpath}")

    with gzip.open(mpath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                names = line[1:].strip().split()
                break

    return  pd.read_csv(mpath, sep="\t", comment="#",  names=names)

def get_meta(config, run_id, sample_name):
    mpath = f"{config.output_dir}/run{run_id}/{sample_name}/meta/{sample_name}.tsv.gz"

    with gzip.open(mpath, "rt") as f:
        for line in f:
            if line.startswith("#"):
                names = ast.literal_eval(line[1:].strip())
                break

    return  pd.read_csv(mpath, sep="\t", comment="#",  names=names)