import argparse
import copy
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.special import logsumexp

def process_snp_phasing(cellsnp_folder, eagle_results_dir, outputfile):
    # create a (snp_id, GT) map from eagle2 output
    snp_gt_map = {}
    
    for c in range(1, 23):
        fname = [
            str(x) for x in Path(eagle_results_dir).glob("*chr{}.phased.vcf.gz".format(c))
        ]
        
        assert len(fname) > 0
        
        fname = fname[0]
        
        tmpdf = pd.read_table(
            fname,
            compression="gzip",
            comment="#",
            sep="\t",
            names=[
                "CHR",
                "POS",
                "ID",
                "REF",
                "ALT",
                "QUAL",
                "FILTER",
                "INFO",
                "FORMAT",
                "PHASE",
            ],
        )
        
        this_snp_ids = [
            "{}_{}_{}_{}".format(c, row.POS, row.REF, row.ALT)
            for i, row in tmpdf.iterrows()
        ]
        
        this_gt = list(tmpdf.iloc[:, -1])
        
        assert len(this_snp_ids) == len(this_gt)
        
        snp_gt_map.update({this_snp_ids[i]: this_gt[i] for i in range(len(this_gt))})
        
    # cellsnp DP (read depth) and AD (alternative allele depth)
    # first get a list of snp_id and spot barcodes
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.base.vcf.gz", header=1, sep="\t")
    snp_list = np.array(
        [
            "{}_{}_{}_{}".format(row["#CHROM"], row.POS, row.REF, row.ALT)
            for i, row in tmpdf.iterrows()
        ]
    )
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.samples.tsv", header=None)
    sample_list = np.array(list(tmpdf.iloc[:, 0]))
    
    # then get the DP and AD matrix
    DP = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.DP.mtx").tocsr()
    AD = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.AD.mtx").tocsr()
    
    # remove SNPs that are not phased
    is_phased = np.array([(x in snp_gt_map) for x in snp_list])
    DP = DP[is_phased, :]
    AD = AD[is_phased, :]
    snp_list = snp_list[is_phased]
    
    # generate a new dataframe with columns (cell, snp_id, DP, AD, CHROM, POS, GT)
    rows, cols = DP.nonzero()
    cell = sample_list[cols]
    snp_id = snp_list[rows]
    DP_df = DP[DP.nonzero()].A.flatten()
    AD_df = AD[DP.nonzero()].A.flatten()
    GT = [snp_gt_map[x] for x in snp_id]
    
    df = pd.DataFrame(
        {
            "cell": cell,
            "snp_id": snp_id,
            "DP": DP_df,
            "AD": AD_df,
            "CHROM": [int(x.split("_")[0]) for x in snp_id],
            "POS": [int(x.split("_")[1]) for x in snp_id],
            "GT": GT,
        }
    )
    
    df.to_csv(
        outputfile, sep="\t", index=False, header=True, compression={"method": "gzip"}
    )
    
    return df


def read_cell_by_snp(allele_counts_file):
    df = pd.read_csv(allele_counts_file, sep="\t", header=0)
    index = np.array([i for i, x in enumerate(df.GT) if x == "0|1" or x == "1|0"])
    df = df.iloc[index, :]
    df.CHROM = df.CHROM.astype(int)
    return df


def cell_by_gene_lefthap_counts(cellsnp_folder, eagle_results_dir, barcode_list):
    # create a (snp_id, GT) map from eagle2 output
    snp_gt_map = {}
    for c in range(1, 23):
        fname = [
            str(x) for x in Path(eagle_results_dir).glob("*chr{}.phased.vcf.gz".format(c))
        ]
        
        assert len(fname) > 0
        
        fname = fname[0]
        tmpdf = pd.read_table(
            fname,
            compression="gzip",
            comment="#",
            sep="\t",
            names=[
                "CHR",
                "POS",
                "ID",
                "REF",
                "ALT",
                "QUAL",
                "FILTER",
                "INFO",
                "FORMAT",
                "PHASE",
            ],
        )
        
        # only keep heterozygous SNPs
        tmpdf = tmpdf[(tmpdf.PHASE == "0|1") | (tmpdf.PHASE == "1|0")]
        this_snp_ids = (
            str(c) + "_" + tmpdf.POS.astype(str) + "_" + tmpdf.REF + "_" + tmpdf.ALT
        ).values
        
        this_gt = tmpdf.PHASE.values
        
        assert len(this_snp_ids) == len(this_gt)
        
        snp_gt_map.update({this_snp_ids[i]: this_gt[i] for i in range(len(this_gt))})

    # cellsnp-lite output
    cellsnp_base = [str(x) for x in Path(cellsnp_folder).glob("cellSNP.base*")][0]
    df_snp = pd.read_csv(
        cellsnp_base,
        comment="#",
        sep="\t",
        names=["tmpCHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
    )
    df_snp["snp_id"] = (
        df_snp.tmpCHR.astype(str)
        + "_"
        + df_snp.POS.astype(str)
        + "_"
        + df_snp.REF
        + "_"
        + df_snp.ALT
    )
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.samples.tsv", header=None)
    sample_list = np.array(list(tmpdf.iloc[:, 0]))
    barcode_mapper = {x: i for i, x in enumerate(sample_list)}
    
    # DP and AD
    DP = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.DP.mtx").tocsr()
    AD = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.AD.mtx").tocsr()
    
    # retain only SNPs that are phased
    is_phased = (df_snp.snp_id.isin(snp_gt_map)).values
    df_snp = df_snp[is_phased]
    df_snp["GT"] = [snp_gt_map[x] for x in df_snp.snp_id]
    DP = DP[is_phased, :]
    AD = AD[is_phased, :]

    # phasing
    phased_AD = np.where((df_snp.GT.values == "0|1").reshape(-1, 1), AD.A, (DP - AD).A)
    phased_AD = scipy.sparse.csr_matrix(phased_AD)

    # re-order based on barcode_list
    index = np.array([barcode_mapper[x] for x in barcode_list if x in barcode_mapper])
    DP = DP[:, index]
    phased_AD = phased_AD[:, index]

    # returned matrix has shape (N_cells, N_snps), which is the transpose of the original matrix
    return (DP - phased_AD).T, phased_AD.T, df_snp.snp_id.values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cellsnplite_results_dir", help="cellsnplite results directory", type=str
    )
    parser.add_argument(
        "-e", "--eagle_results_dir", help="eagle results directory", type=str
    )
    parser.add_argument("-b", "--barcodefile", help="barcode file", type=str)
    parser.add_argument("-o", "--outputdir", help="output directory", type=str)
    
    args = parser.parse_args()

    barcode_list = list(pd.read_csv(args.barcodefile, header=None).iloc[:, 0])
    
    cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids = cell_by_gene_lefthap_counts(
        args.cellsnplite_result_dir, args.eagle_results_dir, barcode_list
    )

    scipy.sparse.save_npz(f"{args.outputdir}/cell_snp_Aallele.npz", cell_snp_Aallele)
    scipy.sparse.save_npz(f"{args.outputdir}/cell_snp_Ballele.npz", cell_snp_Ballele)
    
    np.save(f"{args.outputdir}/unique_snp_ids.npy", unique_snp_ids)
