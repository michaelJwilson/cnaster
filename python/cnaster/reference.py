import pandas as pd


def get_reference_genes(hgtable_file):
    # read gene info and keep only chr1-chr22 and genes appearing in adata
    df_hgtable = pd.read_csv(hgtable_file, header=0, index_col=0, sep="\t")
    df_hgtable = df_hgtable[df_hgtable.chrom.isin([f"chr{i}" for i in range(1, 23)])]

    return df_hgtable
