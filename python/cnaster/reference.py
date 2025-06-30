import pandas as pd


def get_reference_genes(hgtable_file):
    # read gene info and keep only chr1-chr22 and genes appearing in adata
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
    
    # check the chromosome names
    if not ("chr" in str(chr_pos_vector[0][0])):
        df["chrom"] = [int(x[3:]) for x in df.chrom]
        
    df = df.sort_values(by=["chrom", "pos"])

    return df
