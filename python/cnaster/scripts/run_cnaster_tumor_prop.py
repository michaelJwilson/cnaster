import argparse
import copy
import time
import logging

import numpy as np
import pandas as pd
import scipy
import functools
from cnaster.config import YAMLConfig, set_global_config
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf import (
    hmrfmix_concatenate_pipeline,
    merge_by_minspots,
    aggr_hmrf_reassignment,
    hmrf_reassignment_posterior,
    aggr_hmrfmix_reassignment,
    hmrfmix_reassignment_posterior,
    reindex_clones,
)
from cnaster.io import load_input_data
from cnaster.omics import (
    assign_initial_blocks,
    create_bin_ranges,
    form_gene_snp_table,
    get_sitewise_transmat,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnaster.phasing import initial_phase_given_partition
from cnaster.spatial import (
    initialize_clones,
    multislice_adjacency,
    rectangle_initialize_initial_clone,
)
from cnaster.tumor_prop import identify_normal_spots, identify_loh_per_clone, estimator_tumor_proportion
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.neyman_pearson import (
    neyman_pearson_similarity,
    combine_similar_states_across_clones,
)
from cnaster.normal_spot import (
    normal_baf_bin_filter,
    filter_normal_diffexp,
    binned_gene_snp,
)
from cnaster.hmm import pipeline_baum_welch
from cnaster.integer_copy import (
    hill_climbing_integer_copynumber_oneclone,
    hill_climbing_integer_copynumber_fixdiploid,
)
from cnaster.plotting import plot_clones_genomic, plot_clones_spatial

start_time = time.time()


class RuntimeFormatter(logging.Formatter):
    def format(self, record):
        runtime_minutes = (time.time() - start_time) / 60.0
        record.runtime = f"{runtime_minutes:.2f}m"
        return super().format(record)


formatter = RuntimeFormatter(
    fmt="%(asctime)s - %(runtime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler("cnaster.log")
stream_handler = logging.StreamHandler()

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)


def run_cnaster(config_path):
    logger.info("----  Welcome to cnaster  ----")

    config = YAMLConfig.from_file(config_path)

    set_global_config(config)

    # NB start run_parse_n_load::parse_visium::load_joint_data
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
    barcodes = adata.obs.index
    coords = adata.obsm["X_pos"]

    sample_list = [adata.obs["sample"].iloc[0]]

    for i in range(1, adata.shape[0]):
        if adata.obs["sample"].iloc[i] != sample_list[-1]:
            sample_list.append(adata.obs["sample"].iloc[i])

    logger.info(f"Found {len(sample_list)} unique samples, e.g. {sample_list[:3]}")

    # NB assign index to unique sample names.
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

    logger.info(f"Forming gene & snp meta data.")

    # NB parse_visium::combine_gene_snps
    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )

    logger.info(f"Assigning initial blocks")

    # NB parse_visium::create_haplotype_block_ranges
    df_gene_snp = assign_initial_blocks(
        df_gene_snp, adata, cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids
    )

    logger.info(f"Summarizing counts for blocks")

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

    # NB 1D array / list?
    log_sitewise_transmat = get_sitewise_transmat(
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    # NB equivalent to parse_visium::perform_partition
    # TODO (requires paste).
    initial_clone_for_phasing = initialize_clones(
        coords,
        sample_ids,
        x_part=config.phasing.npart_phasing,
        y_part=config.phasing.npart_phasing,
    )

    logger.warning("Assuming 5 BAF states for phasing.")

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
        1.0e-3,  # MAGIC tol on HMM parameter end.
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(
        f"Solved for initial phase given Eagle & BAF in {(time.time() - start_time):.2f} seconds."
    )

    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

    logger.info(f"Recalculating blocks given new phasing")

    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        single_total_bb_RD,
        refined_lengths,
        config.quality.secondary_min_umi,
    )

    logger.info(f"Recalculating counts given new blocks")

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

    logger.info("Solving for multislice_adjaceny.")

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

    # DEPRECATE
    # n_pooled = np.median(np.sum(smooth_mat > 0, axis=0).A.flatten())

    # NB end run_parse_n_load::parse_visium.
    # TODO table_bininfo? table_rdrbaf? table_meta?

    # TODO
    copy_single_X_rdr = copy.copy(single_X[:, 0, :])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)

    # NB baf-only run;
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    logger.warning(
        "Solving for tumor prop. with MAGICs: n_states_for_tumorprop, n_clones_for_tumorprop, etc."
    )

    n_states_for_tumorprop = 5  # MAGIC
    n_clones_for_tumorprop = 3  # MAGIC
    n_rdrclones_for_tumorprop = 3  # MAGIC
    max_outer_iter_for_tumorprop = 10  # MAGIC
    max_iter_for_tumorprop = 20  # MAGIC
    MIN_PROP_UNCERTAINTY = 0.05  # MAGIC

    logger.info("Solving for multislice_adjaceny.")

    # TODO config.hmrf.n_clones -> n_clones_for_tumorprop
    initial_clone_index = rectangle_initialize_initial_clone(
        coords, n_clones_for_tumorprop, random_state=0
    )

    logger.info("Solving HMM+HMRF for copy state and clones with BAF only.")

    res = hmrfmix_concatenate_pipeline(
        None,
        None,
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        None,
        initial_clone_index,
        n_states_for_tumorprop,  # TODO
        log_sitewise_transmat,
        smooth_mat=smooth_mat,
        adjacency_mat=adjacency_mat,
        sample_ids=sample_ids,
        max_iter_outer=max_outer_iter_for_tumorprop,
        nodepotential=config.hmrf.nodepotential,
        hmmclass=hmm_nophasing,
        params="sp",
        t=config.hmm.t,
        random_state=config.hmm.gmm_random_state,
        fix_NB_dispersion=config.hmm.fix_NB_dispersion,
        shared_NB_dispersion=config.hmm.shared_NB_dispersion,
        fix_BB_dispersion=config.hmm.fix_BB_dispersion,
        shared_BB_dispersion=config.hmm.shared_BB_dispersion,
        is_diag=True,
        max_iter=max_iter_for_tumorprop,  # TODO
        tol=config.hmm.tol,
        spatial_weight=config.hmrf.spatial_weight,
        tumorprop_threshold=config.hmrf.tumorprop_threshold,
    )

    n_obs = single_X.shape[0]

    # NB 2.5 mins.
    merging_groups, merged_res = merge_by_minspots(
        res["new_assignment"],
        res,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=n_obs * config.hmrf.min_avgumi_per_clone,
    )

    # TODO CHECK
    n_baf_clones = len(merging_groups)
    combined_assignment = copy.copy(merged_res["new_assignment"])
    offset_clone = 0
    combined_p_binom = []
    offset_state = 0
    combined_pred_cnv = []

    clone_res = {}

    logger.info(
        f"Refinining {n_baf_clones} BAF identified clones with RDR data assuming n_clones_rdr={n_rdrclones_for_tumorprop}"
    )

    # TODO HACK?
    single_X[:, 0, :] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean

    logger.warning(f"Adding back RDR; neglected in CalicoST?")
    
    for bafc in range(n_baf_clones):
        logger.info(f"Solving for BAF clone {bafc}/{n_baf_clones}.")

        prefix = f"clone{bafc}"
        idx_spots = np.where(merged_res['new_assignment'] == bafc)[0]

        # NB minimum B allele read count on pseudobulk to split clones.
        if np.sum(single_total_bb_RD[:, idx_spots]) < single_X.shape[0] * 50:  # MAGIC
            combined_assignment[idx_spots] = offset_clone
            offset_clone += 1
            combined_p_binom.append(merged_res["new_p_binom"])
            combined_pred_cnv.append(merged_res["pred_cnv"] + offset_state)
            offset_state += merged_res["new_p_binom"].shape[0]
            continue

        # TODO tumor_prop, i.e. _mix.
        initial_clone_index = rectangle_initialize_initial_clone(
            coords[idx_spots],
            n_rdrclones_for_tumorprop,  # TODO
            random_state=0,  # TODO HACK.
        )

        initial_assignment = np.zeros(len(idx_spots), dtype=int)

        for c, idx in enumerate(initial_clone_index):
            initial_assignment[idx] = c

        # NB
        clone_res[prefix] = {
            "barcodes": barcodes[idx_spots],
            "num_iterations": 0,
            "round-1_assignment": initial_assignment,
        }

        # HMRF + HMM using RDR data.
        copy_slice_sample_ids = copy.copy(sample_ids[idx_spots])

        clone_res[prefix] = clone_res[prefix] | hmrfmix_concatenate_pipeline(
            None,
            None,
            single_X[:, :, idx_spots],
            lengths,
            single_base_nb_mean[:, idx_spots],
            single_total_bb_RD[:, idx_spots],
            single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
            initial_clone_index,  # NB
            n_states=n_states_for_tumorprop,  # TODO
            log_sitewise_transmat=log_sitewise_transmat,
            smooth_mat=smooth_mat[np.ix_(idx_spots, idx_spots)],
            adjacency_mat=adjacency_mat[np.ix_(idx_spots, idx_spots)],
            sample_ids=copy_slice_sample_ids,
            max_iter_outer=10,  # TODO MAGIC
            nodepotential=config.hmrf.nodepotential,
            hmmclass=hmm_nophasing,
            params="smp",
            t=config.hmm.t,
            random_state=config.hmm.gmm_random_state,
            fix_NB_dispersion=config.hmm.fix_NB_dispersion,
            shared_NB_dispersion=config.hmm.shared_NB_dispersion,
            fix_BB_dispersion=config.hmm.fix_BB_dispersion,
            shared_BB_dispersion=config.hmm.shared_BB_dispersion,
            is_diag=True,
            max_iter=max_iter_for_tumorprop,  # TODO MAGIC
            tol=config.hmm.tol,
            spatial_weight=config.hmrf.spatial_weight,
            tumorprop_threshold=config.hmrf.tumorprop_threshold,
        )

        combined_assignment[idx_spots] = clone_res["new_assignment"] + offset_clone
        offset_clone += 1 + np.max(clone_res["new_assignment"])
        combined_p_binom.append(clone_res["new_p_binom"])
        combined_pred_cnv.append(clone_res["pred_cnv"] + offset_state)
        offset_state += clone_res["new_p_binom"].shape[0]

    combined_p_binom = np.vstack(combined_p_binom)
    combined_pred_cnv = np.concatenate(combined_pred_cnv)

    """
    normal_candidate = identify_normal_spots(
        single_X,
        single_total_bb_RD,
        merged_res["new_assignment"],
        merged_res["pred_cnv"],
        merged_res["new_p_binom"],
        min_count=single_X.shape[0] * 200, # TODO
    )
    loh_states, is_B_lost, rdr_values, clones_hightumor = identify_loh_per_clone(
        single_X,
        combined_assignment,
        combined_pred_cnv,
        combined_p_binom,
        normal_candidate,
        single_total_bb_RD,
    )
    assignments = pd.DataFrame(
        {"coarse": merged_res["new_assignment"], "combined": combined_assignment}
    )
    
    # NB pool across adjacent spot to increase the UMIs covering LOH region.
    _, tp_smooth_mat = multislice_adjacency(
        sample_ids,
        sample_list,
        coords,
        single_total_bb_RD,
        exp_counts,
        across_slice_adjacency_mat=None,
        construct_adjacency_method=config["construct_adjacency_method"],
        maxspots_pooling=7,
        construct_adjacency_w=config["construct_adjacency_w"],
    )
    single_tumor_prop, _ = estimator_tumor_proportion(
        single_X,
        single_total_bb_RD,
        assignments,
        combined_pred_cnv,
        loh_states,
        is_B_lost,
        rdr_values,
        clones_hightumor,
        smooth_mat=tp_smooth_mat,
    )
    
    # NB post-processing to remove negative tumor proportions
    single_tumor_prop = np.where(
        single_tumor_prop < MIN_PROP_UNCERTAINTY,
        MIN_PROP_UNCERTAINTY,
        single_tumor_prop,
    )
    single_tumor_prop[normal_candidate] = 0
    
    # NB save single_tumor_prop to file
    pd.DataFrame({"Tumor": single_tumor_prop}, index=barcodes).to_csv(
        f"{config['output_dir']}/loh_estimator_tumor_prop.tsv", header=True, sep="\t"
    )
    """

    logger.info(f"Done in {(time.time() - start_time)/60.:.2f} minutes.")


# NB run_cnaster config.yaml
def main():
    parser = argparse.ArgumentParser(description="Run CNAster pipeline")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file",
    )

    args = parser.parse_args()

    run_cnaster(args.config_path)


if __name__ == "__main__":
    main()
