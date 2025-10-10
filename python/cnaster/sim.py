import scipy
import pandas as pd
import numpy as np
from cnaster.config import get_global_config


def load_tables_to_matrices():
    """
    Load tables and adjacency from parse_visium_joint or parse_visium_single, and convert to HMM input matrices.
    """
    config = get_global_config()

    table_bininfo = pd.read_csv(
        f"{config.paths.output_dir}/parsed_inputs/table_bininfo.csv.gz",
        header=0,
        index_col=None,
        sep="\t",
    )
    table_rdrbaf = pd.read_csv(
        f"{config.paths.output_dir}/parsed_inputs/table_rdrbaf.csv.gz",
        header=0,
        index_col=None,
        sep="\t",
    )
    table_meta = pd.read_csv(
        f"{config.paths.output_dir}/parsed_inputs/table_meta.csv.gz",
        header=0,
        index_col=None,
        sep="\t",
    )
    adjacency_mat = scipy.sparse.load_npz(
        f"{config.paths.output_dir}/parsed_inputs/adjacency_mat.npz"
    )
    smooth_mat = scipy.sparse.load_npz(
        f"{config.paths.output_dir}/parsed_inputs/smooth_mat.npz"
    )
    #
    df_gene_snp = pd.read_csv(
        f"{config.paths.output_dir}/parsed_inputs/gene_snp_info.csv.gz",
        header=0,
        index_col=None,
        sep="\t",
    )
    df_gene_snp = df_gene_snp.replace(np.nan, None)

    n_spots = table_meta.shape[0]
    n_bins = table_bininfo.shape[0]

    # construct single_X
    single_X = np.zeros((n_bins, 2, n_spots))
    single_X[:, 0, :] = table_rdrbaf["EXP"].values.reshape((n_bins, n_spots), order="F")
    single_X[:, 1, :] = table_rdrbaf["B"].values.reshape((n_bins, n_spots), order="F")

    # construct single_base_nb_mean, lengths
    single_base_nb_mean = (
        table_bininfo["NORMAL_COUNT"].values.reshape(-1, 1)
        / np.sum(table_bininfo["NORMAL_COUNT"].values)
        @ np.sum(single_X[:, 0, :], axis=0).reshape(1, -1)
    )

    # construct single_total_bb_RD
    single_total_bb_RD = table_rdrbaf["TOT"].values.reshape(
        (n_bins, n_spots), order="F"
    )

    # construct log_sitewise_transmat
    log_sitewise_transmat = table_bininfo["LOG_PHASE_TRANSITION"].values

    # construct bin info and lengths and x_gene_list
    df_bininfo = table_bininfo
    lengths = np.array(
        [np.sum(table_bininfo.CHR == c) for c in df_bininfo.CHR.unique()]
    )

    # construct barcodes
    barcodes = table_meta["BARCODES"]

    # construct coords
    coords = table_meta[["X", "Y"]].values

    # construct single_tumor_prop
    single_tumor_prop = (
        table_meta["TUMOR_PROPORTION"].values
        if "TUMOR_PROPORTION" in table_meta.columns
        else None
    )

    # construct sample_list and sample_ids
    sample_list = [table_meta["SAMPLE"].values[0]]
    for i in range(1, table_meta.shape[0]):
        if table_meta["SAMPLE"].values[i] != sample_list[-1]:
            sample_list.append(table_meta["SAMPLE"].values[i])
    sample_ids = np.zeros(table_meta.shape[0], dtype=int)

    for s, sname in enumerate(sample_list):
        index = np.where(table_meta["SAMPLE"].values == sname)[0]
        sample_ids[index] = s

    # expression UMI count matrix
    exp_counts = pd.read_pickle(
        f"{config.paths.output_dir}/parsed_inputs/exp_counts.pkl"
    )

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        df_bininfo,
        df_gene_snp,
        barcodes,
        coords,
        single_tumor_prop,
        sample_list,
        sample_ids,
        adjacency_mat,
        smooth_mat,
        exp_counts,
    )
