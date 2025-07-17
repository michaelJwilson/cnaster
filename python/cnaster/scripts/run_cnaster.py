import copy
import scipy
import logging
import numpy as np
import pandas as pd

from cnaster.config import YAMLConfig
from cnaster.io import load_input_data
from cnaster.omics import (
    form_gene_snp_table,
    assign_initial_blocks,
    summarize_counts_for_blocks,
    get_sitewise_transmat,
    create_bin_ranges,
    summarize_counts_for_bins,
)

from cnaster.spatial import initialize_clones, multislice_adjacency
from cnaster.phasing import initial_phase_given_partition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    # HACK
    config_path = f"/u/mw9568/research/repos/cnaster/config_turing.yaml"
    config = YAMLConfig.from_file(config_path)

    (
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        across_slice_adjacency_mat,
    ) = load_input_data(
        config,
        filter_gene_file=config.references.filtergenelist_file,
        filter_range_file=config.references.filterregion_file,
    )

    # TODO CHECK
    coords = adata.obsm["X_pos"]

    sample_list = [adata.obs["sample"].iloc[0]]

    for i in range(1, adata.shape[0]):
        if adata.obs["sample"].iloc[i] != sample_list[-1]:
            sample_list.append(adata.obs["sample"].iloc[i])

    # convert sample name to index
    sample_ids = np.zeros(adata.shape[0], dtype=int)

    for s, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        sample_ids[index] = s

    single_tumor_prop = None

    """
    # TODO
    if config.preprocessing.tumorprop_file is not None:
        df_tumorprop = pd.read_csv(
            config.preprocessing.tumorprop_file, sep="\t", header=0, index_col=0
        )
        df_tumorprop = df_tumorprop[["Tumor"]]
        df_tumorprop.columns = ["tumor_proportion"]

        adata.obs = adata.obs.join(df_tumorprop)

        single_tumor_prop = adata.obs["tumor_proportion"]
    """

    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )

    df_gene_snp = assign_initial_blocks(
        df_gene_snp, adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids
    )

    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
    ) = summarize_counts_for_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
    )

    log_sitewise_transmat = get_sitewise_transmat(
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    initial_clone_for_phasing = initialize_clones(
        coords,
        sample_ids,
        x_part=config.phasing.npart_phasing,
        y_part=config.phasing.npart_phasing,
    )

    # TODO updates mu? as initialization?
    phase_indicator, refined_lengths = initial_phase_given_partition(
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        initial_clone_for_phasing,
        5,  # MAGIC n_states
        log_sitewise_transmat,
        "sp",  # MAGIC params (start prob. and baf states).
        config.hmm.t_phaseing,
        config.hmm.gmm_random_state,
        config.hmm.fix_NB_dispersion,
        config.hmm.shared_NB_dispersion,
        config.hmm.fix_BB_dispersion,
        config.hmm.shared_BB_dispersion,
        config.hmm.max_iter,  # MAGIC max_iter
        1.0e-3,  # MAGIC tol
        threshold=config.hmrf.tumorprop_threshold,
    )

    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        single_total_bb_RD,
        refined_lengths,
        config.quality.secondary_min_umi,
    )

    # TODO separate transmat.
    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
    ) = summarize_counts_for_bins(
        df_gene_snp,
        adata,
        single_X,
        single_total_bb_RD,
        phase_indicator,
        nu=config.phasing.nu,
        logphase_shift=config.phasing.logphase_shift,
        geneticmap_file=config.references.geneticmap_file,
    )

    # NB expression count dataframe
    exp_counts = pd.DataFrame.sparse.from_spmatrix(
        scipy.sparse.csc_matrix(adata.layers["count"]),
        index=adata.obs.index,
        columns=adata.var.index,
    )

    # NB smooth & adjacency matrix for each sample
    adjacency_mat, smooth_mat = multislice_adjacency(
        sample_ids,
        sample_list,
        coords,
        single_total_bb_RD,
        exp_counts,
        across_slice_adjacency_mat,
        construct_adjacency_method=config.hmrf.construct_adjacency_method,
        maxspots_pooling=config.hmrf.maxspots_pooling,
        construct_adjacency_w=config.hmrf.construct_adjacency_w,
    )

    n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())

    # TODO
    copy_single_X_rdr = copy.copy(single_X[:, 0, :])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)

    # NB baf-only run;
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    initial_clone_index = rectangle_initialize_initial_clone(
        coords, config["n_clones"], random_state=r_hmrf_initialization
    )

    logger.info("Done.\n\n")
