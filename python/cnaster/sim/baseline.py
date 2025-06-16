import logging
import numpy as np
import pandas as pd
from pathlib import Path
import gzip

logger = logging.getLogger(__name__)

def generate_exp_baseline(config):
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp
    exp_gene_segment = config.exp_gene_kbp * config.segment_size_kbp

    lambdas = np.random.poisson(lam=exp_gene_segment, size=num_segments)
    lambdas = lambdas / lambdas.sum()

    outdir = Path(config.output_dir) / "baseline"
    outdir.mkdir(parents=True, exist_ok=True)

    opath = outdir / "expression.tsv.gz"

    df = pd.DataFrame({
        "segment": np.arange(num_segments),
        "expression": lambdas
    })

    logger.info(f"Writing normalized baseline expression to {opath}")

    with gzip.open(opath, "wt") as f:
        f.write("# segment\tbaseline\n")
        df.to_csv(f, sep="\t", index=False, header=False)

def generate_snp_baseline(config):
    num_segments = config.mappable_genome_kbp // config.segment_size_kbp
    exp_snps_segment = config.exp_snp_kbp * config.segment_size_kbp

    num_snps_segments = np.random.poisson(lam=exp_snps_segment, size=num_segments)

    # NB write
    outdir = Path(config.output_dir) / "baseline"
    outdir.mkdir(parents=True, exist_ok=True)

    opath = outdir / "snps.tsv.gz"

    df = pd.DataFrame({
        "segment": np.arange(num_segments),
        "snps": num_snps_segments
    })

    logger.info(f"Writing SNP baseline to {opath}")

    with gzip.open(opath, "wt") as f:
        f.write("# snp\tumi\n")
        df.to_csv(f, sep="\t", index=False, header=False)   