import logging
import re

import pandas as pd
import polars as pl
import pyranges as pr
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


cancer_gene_patterns = (
    r"^MT-",  # Mitochondrial genes
    r"^S100A6$",  # Exact match for S100A6
    r"^B2M$",  # Exact match for B2M
    r"^HLA-B$",  # Exact match for HLA-B
    r"^TPT1$",  # Exact match for TPT1
    r"^FTL$",  # Exact match for FTL
    r"^FTH1$",  # Exact match for FTH1
    r"^RPL",  # Ribosomal protein L genes
    r"^RPS",  # Ribosomal protein S genes
)

pl.Config.set_tbl_cols(-1)


def exp_cancer_gene(gene_name):
    return any(re.match(pattern, gene_name) for pattern in cancer_gene_patterns)


def get_reference_genes(hgtable_file):
    config = get_global_config()

    # TODO HACK use legacy only for now.
    if True or config.run.legacy:
        logger.info_once(f"Assuming legacy gene annotation={hgtable_file}.")

        # NB read gene info and keep only chr1-chr22 and genes appearing in adata
        #    name2  chrom  cdsStart    cdsEnd
        df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
        df_hgtable = df_hgtable[
            df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])
        ]
        df_hgtable = df_hgtable.rename(
            columns={
                "chrom": "Chromosome",
                "cdsStart": "Start",
                "cdsEnd": "End",
                "name2": "gene",
            }
        )
    else:
        # TODO rename_attr=True
        df_hgtable = pr.read_gtf(config.references.annotation_file, full=True)
        df_hgtable = df_hgtable.query("Feature == 'gene'").drop_duplicates(
            "gene_id", keep="first"
        )
        df_hgtable = df_hgtable[["Chromosome", "Start", "End", "gene_name", "gene_id"]]

        # NB drop .5 suffix to gene_id.
        df_hgtable["gene_id"] = df_hgtable["gene_id"].str.replace(
            r"\.\d+$", "", regex=True
        )
        df_hgtable = df_hgtable[
            df_hgtable.Chromosome.isin([f"chr{i}" for i in range(1, 23)])
        ]
        df_hgtable = df_hgtable.rename(columns={"gene_name": "gene"})

    logger.info(f"Read reference genes:\n{df_hgtable}")

    df_gene = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.Chromosome.to_numpy()],
            "START": df_hgtable.Start.to_numpy(),
            "END": df_hgtable.End.to_numpy(),
            "snp_id": None,
            "gene": df_hgtable.gene.to_numpy(),
            "is_interval": True,
        }
    )

    # df_gene["LENGTH"] = df_gene["END"] - df_gene["START"]

    return df_gene


def get_reference_recomb_rates(geneticmap_file):
    """
    Attributes
    ----------
    chr_pos_vector : list of pairs
        list of (chr, pos) pairs of SNPs
    """
    df = pd.read_csv(geneticmap_file, header=0, sep="\t")
    df = df[df.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    # NB drop "chr" prefix.
    df["chrom"] = df.chrom.str.replace("chr", "")

    df = df.sort_values(by=["chrom", "pos"])

    logger.info_once(
        f"Read reference recombination rates from {geneticmap_file}:\n{df.head()}"
    )

    return df
