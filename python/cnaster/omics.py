import pandas as pd
from cnaster.reference import get_reference_genes

# TODO place elsewhere.                                                                                                                                                                                                                                                                             
def form_gene_snp_table(unique_snp_ids, hgtable_file, adata):
    # read gene info and keep only chr1-chr22 and genes appearing in adata                                                                                                                                                                                                                          
    df_hgtable = get_reference_genes(hgtable_file)
    df_hgtable = df_hgtable[df_hgtable.name2.isin(adata.var.index)]

    # a data frame including both gene and SNP info: CHR, START, END, snp_id, gene, is_interval                                                                                                                                                                                                      
    df_gene_snp = pd.DataFrame(
        {
            "CHR": [int(x[3:]) for x in df_hgtable.chrom.values],
            "START": df_hgtable.cdsStart.values,
            "END": df_hgtable.cdsEnd.values,
            "snp_id": None,
            "gene": df_hgtable.name2.values,
            "is_interval": True,
        }
    )

    # add SNP info                                                                                                                                                                                                                                                                                   
    snp_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids])

    df_gene_snp = pd.concat(
        [
            df_gene_snp,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_pos + 1,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )

    df_gene_snp.sort_values(by=["CHR", "START"], inplace=True)
    
    # assign genes to each SNP.                                                                                                                                                                                                                                                                     
    # for each SNP (with not null snp_id), find the previous gene (is_interval == True) such that the SNP start position is within the gene start and end interval                                                                                                                                   
    vec_is_interval = df_gene_snp.is_interval.values

    vec_chr = df_gene_snp.CHR.values
    vec_start = df_gene_snp.START.values
    vec_end = df_gene_snp.END.values

    for i in np.where(df_gene_snp.gene.isnull())[0]:
        # TODO first SNP has no gene.                                                                                                                                                                                                                                                               
        if i == 0:
            continue

        this_pos = vec_start[i]
        j = i - 1

        # TODO overlapping genes?                                                                                                                                                                                                                                                                   
        while j >= 0 and j >= i - 50 and vec_chr[i] == vec_chr[j]:
            if (
                vec_is_interval[j]
                and vec_start[j] <= this_pos
                and vec_end[j] > this_pos
            ):
                df_gene_snp.iloc[i, 4] = df_gene_snp.iloc[j]["gene"]
                break

            j -= 1

    # remove SNPs that have no corresponding genes                                                                                                                                                                                                                                                  
    df_gene_snp = df_gene_snp[~df_gene_snp.gene.isnull()]

    return df_gene_snp
