import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_reference_genes(hgtable_file):
    # NB read gene info and keep only chr1-chr22 and genes appearing in adata
    df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
    df_hgtable = df_hgtable[df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    return df_hgtable


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

    logger.info(
        f"Read reference recombination rates from {geneticmap_file}:\n{df.head()}"
    )

    return df
