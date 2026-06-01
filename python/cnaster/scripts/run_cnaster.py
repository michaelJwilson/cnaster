import argparse
import copy
import time
import random
import scipy

import numpy as np
import pandas as pd
import scipy
import functools
from cnaster.config import start_time, YAMLConfig, set_global_config
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf import (
    hmrfmix_concatenate_pipeline,
    merge_by_minspots,
    aggr_hmrf_reassignment,
    # hmrf_reassignment_posterior,
    aggr_hmrfmix_reassignment,
    # hmrfmix_reassignment_posterior,
    reindex_clones,
)
from cnaster.hmrf_utils import get_clone_indices, get_clone_assignment
from cnaster.io import load_input_data, get_sample_list, read_tumor_prop
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
    # fixed_rectangle_partition,
    best_equal_partition,
    initialize_clones,
    multislice_adjacency,
    rectangle_initialize_initial_clone,
    # sufficient_umis_initial_clone,
)
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.neyman_pearson import (
    neyman_pearson_similarity,
    combine_similar_states_across_clones,
)
from cnaster.normal_spot import (
    determine_normal_candidates,
    determine_normal_baseline,
    normal_baf_bin_filter,
    filter_normal_diffexp,
    binned_gene_snp,
)
from numba import njit

# from cnaster.sim import load_tables_to_matrices
from cnaster.hmm import pipeline_baum_welch
# from cnaster.hmm_initialize import plot_cna_mixture
from cnaster.utils import (
    configure_output_dir,
    merge_dicts,
    write_tsv,
    write_fig,
    pause,
)
from cnaster.integer_copy import (
    hill_climbing_integer_copynumber_oneclone,
    hill_climbing_integer_copynumber_fixdiploid,
)
from cnaster.logger import get_logger
from cnaster.plotting import (
    plot_clones_spatial,
    # plot_gene_snp_spatial,
    plot_adjacency,
    # plot_recombination_rates,
    plot_copy_states,
)
from cnaster.plot_genomic import plot_clones_genomic_raw, plot_clones_genomic
from cnaster.annotation import get_clone_label_annotation

# from cnaster.reference import get_reference_recomb_rates
# from cnaster.perturb import perturb_phase
# from cnaster.hmm_utils import get_em_solver_params


@njit
def set_numba_seed(value):
    np.random.seed(value)


logger = get_logger(__name__, start_time=start_time)


def run_cnaster(config_path, over_rides=None):
    logger.info("----  Welcome to cna-maste  ----")
    
    config = YAMLConfig.from_file(config_path)
    config.over_ride(over_rides)
    config.issue_warnings()

    logger.info(f"Read configuration:\n{config}")

    set_global_config(config)

    output_dir, plots_dir = configure_output_dir(config)

    logger.info(f"Set (numpy) random seed={config.hmrf.random_state}")

    # TODO fix reproducibility - set random seed globally.
    np.random.seed(config.hmrf.random_state)
    random.seed(config.hmrf.random_state)
    set_numba_seed(config.hmrf.random_state)

    # NB legacy simulated data loading - generates matrices for all steps of the pipeline.
    # (
    #     lengths,
    #     single_X,
    #     single_base_nb_mean,
    #     single_total_bb_RD,
    #     log_sitewise_transmat,
    #     df_bininfo,
    #     df_gene_snp,
    #     barcodes,
    #     coords,
    #     single_tumor_prop,
    #     sample_list,
    #     sample_ids,
    #     adjacency_mat,
    #     smooth_mat,
    #     exp_counts,
    # ) = load_tables_to_matrices()

    # original_single_X = single_X.copy()

    # TODO HACK check against above.
    # smooth_mat, adjacency_mat = choose_adjacency_by_readcounts(
    #     coords, single_total_bb_RD
    # )
    # smooth_mat.eliminate_zeros()
    # adjacency_mat.eliminate_zeros()

    # logger.info(f"Found adjacency matrix:\n{adjacency_mat}")

    # NB renormalize cumulative edge weight to median in each case; as corners are under-weighted.
    # adjacency_mat = renormalize_adjacency_mat(adjacency_mat)
    
    # NB start equivalent to run_parse_n_load::parse_visium::load_joint_data
    #
    #    adata: (barcode x gene) transcripts ('count') + 'tumor_annotation' + 'X_pos' + slice ('sample').
    #    cell_snp_Aallele: haplotype h0 counts (barcode x snp).
    #    cell_snp_Ballele: haplotype h1 counts (barcode x snp).
    #
    #    DEPRECATE
    #    unique_snp_ids: {contig}_{pos}_{R}_{A} for all snps.
    (
        coords,
        barcodes,
        adata,
        exp_counts,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        across_slice_adjacency_mat,
    ) = load_input_data(
        config,
        filter_gene_file=config.references.filtergenelist_file,
        filter_range_file=config.references.filterregion_file,
        min_snp_umis=config.quality.spot_min_snp_umis,
        min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
    )

    pause()

    # NB sample list derived from adata.obs['sample'] - removes adjacent duplicates.
    #    sample_ids: unique enum for each entry in sample_list.  One per adata.obs entry.
    sample_list, sample_ids = get_sample_list(adata)
    single_tumor_prop = read_tumor_prop(adata, config=config)

    # NB parse_visium::combine_gene_snps
    #    [ chr, start, end, snp_id, gene, is_interval (is_gene) ]
    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )

    pause()

    # NB parse_visium::create_haplotype_block_ranges
    df_gene_snp = assign_initial_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        initial_min_umi=config.quality.phasing_min_snp_umis,
    )

    pause()

    # TODO utilize <BLOCK COUNTS>
    # NB num. of blocks per contig;
    #    snp-based H0 and H0+H1 counts block;
    #    total umis per block.
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

    pause()

    # NB 1D array of expected phase error rate.
    log_sitewise_transmat = get_sitewise_transmat(
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    # NB pseudobulk formed of all spots.                                                                                                                                                                   
    initial_clone_pseudobulk = [[ii for ii in range(len(coords))]]
    pseudobulk_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_pseudobulk,
        lengths,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
    )

    fig_path = f"{plots_dir}/pseudobulk_clones_genomic.pdf"
    write_fig(
        fig_path, pseudobulk_clones_genomic, transparent=True, bbox_inches="tight"
    )

    pause()

    #
    # ==================================================
    # baf-derived phasing (assuming initial clones)
    # ==================================================
    # 

    # NB known clone annotation per spot.
    if config.annotation.clone_label is not None:
        initial_clone_index_baf, _ = get_clone_label_annotation(config)

        # NB assumes the known clone labels for phasing.
        initial_clone_for_phasing = initial_clone_index_baf
    else:
        # NB reference assignment, not a copy?
        initial_clone_index_baf = None

        # NB  rectangular partition across multiple slices.
        #     equivalent to parse_visium::perform_partition
        initial_clone_for_phasing = initialize_clones(
            coords,
            sample_ids,  # NB for all spots in all slices.
            x_part=config.phasing.npart_phasing,
            y_part=config.phasing.npart_phasing,
        )

    if "he_label" in adata.obsm:
        logger.info(f"Refining initial clone partition with h&e derived segmentation.")

        spatial_assignment = get_clone_assignment(coords, initial_clone_for_phasing)

        # NB per-spot h&e label derived from gray-scale percentiles.
        he_assignment = adata.obsm["he_label"].flatten()
        
        clone_assignment, _ = pd.factorize(
            list(zip(he_assignment, spatial_assignment))
        )
        
        initial_clone_for_phasing = initial_clone_index_baf = get_clone_indices(
            clone_assignment, np.unique(clone_assignment),
        )

    assignment = get_clone_assignment(coords, initial_clone_for_phasing)
    assignment = pd.Series([f"clone {x}" for x in assignment])

    phasing_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    # NB plot of the clones assumed for initial phasing.
    fig_path = f"{plots_dir}/phasing_clones_spatial.pdf"
    write_fig(fig_path, phasing_clones_fig, transparent=True, bbox_inches="tight")
    
    prephasing_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_for_phasing,
        lengths,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
    )

    fig_path = f"{plots_dir}/prephasing_clones_genomic.pdf"
    write_fig(
        fig_path, prephasing_clones_genomic, transparent=True, bbox_inches="tight"
    )

    if config.phasing.run:
        if config.run.legacy:
            logger.warning("Assuming (magic) five baf states for phasing.")
            n_states_phasing = 5
        else:
            n_states_phasing = config.hmm.n_states

        # NB single_base_nb_mean initialized to zero - requires normal spot. determination.
        _, phase_indicator, refined_lengths = initial_phase_given_partition(
            single_X,
            lengths,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            initial_clone_for_phasing,
            n_states_phasing,
            log_sitewise_transmat,
            "sp",  # MAGIC params (start prob. & baf states, no transition).
            config.hmm.t_phaseing,
            config.hmm.gmm_random_state,
            config.hmm.fix_NB_dispersion,
            config.hmm.shared_NB_dispersion,
            config.hmm.fix_BB_dispersion,
            config.hmm.shared_BB_dispersion,
            config.hmm.max_iter,
            1.0e-3,  # MAGIC tol on HMM parameter end.
            threshold=config.hmrf.tumorprop_threshold,
        )

        logger.info(
            f"Solved for initial phase given Eagle & BAF in {(time.time() - start_time):.2f} seconds."
        )
    else:
        phase_indicator = np.zeros(single_X.shape[0])
        refined_lengths = lengths

    # NB phase is None for genes and otherwise 0/1 for baf-inferred phase.
    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

    # NB generates new genomic intervals ("bin_id") based on baf-derived phasing.
    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        single_X,
        single_total_bb_RD,
        refined_lengths,
        config.quality.secondary_min_umi,
        config.quality.secondary_min_snp_umi,
        config.quality.secondary_min_normal_umi,
        max_binlength=config.quality.max_binlength,
    )

    pause()

    logger.info(f"Recalculating counts given new baf-phasing intervals.")

    # TODO separate transmat.
    # NB counters per baf-phasing derived genomic intervals.
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

    pause()

    # TODO copy rename.
    postphasing_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_for_phasing,
        lengths,
        res=None,  # TODO e.g. plot phasing inferred states.
        single_tumor_prop=None,
        sample_list=sample_list,
    )

    fig_path = f"{plots_dir}/postphasing_clones_genomic.pdf"
    write_fig(
        fig_path, postphasing_clones_genomic, transparent=True, bbox_inches="tight"
    )

    pseudobulk_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_pseudobulk,
        lengths,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
    )

    fig_path = f"{plots_dir}/postphasing_pseudobulk_clones_genomic.pdf"
    write_fig(
        fig_path, pseudobulk_clones_genomic, transparent=True, bbox_inches="tight"
    )

    pause()

    # NB smooth pooling matrix & distance based (exponential decay) adjacency.
    adjacency_mat, smooth_mat = multislice_adjacency(
        sample_ids,
        sample_list,
        coords,
        single_total_bb_RD, # NEGLECTED?
        exp_counts, # NEGLECTED?
        across_slice_adjacency_mat,
        construct_adjacency_method=config.hmrf.construct_adjacency_method,
        maxspots_pooling=config.hmrf.maxspots_pooling,
        construct_adjacency_w=config.hmrf.construct_adjacency_w,
        unit_xsquared=config.hmrf.unit_xsquared,  # TODO
        unit_ysquared=config.hmrf.unit_ysquared,  # TODO
    )

    adjacency_fig = plot_adjacency(
        coords,
        smooth_mat,
        adjacency_mat,
        pointsize=5,
        base_height=6,
        sample_list=sample_list,
    )

    # fig_path = f"{plots_dir}/adjacency.pdf"
    # write_fig(fig_path, adjacency_fig, transparent=True, bbox_inches="tight")
    # NB end run_parse_n_load::parse_visium.

    pause()

    # NB by construction, require normal spots (based on BAF to determine baseline).
    assert np.all(single_base_nb_mean == 0)

    # TODO UGH HACK
    copy_single_X_rdr = copy.copy(single_X[:, 0, :])

    # NB zeros
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)

    logger.info(
        f"Assuming initial clone configuration for baf-inferred clones & copy states."
    )

    # TODO HACK? adata.layers["count"]
    if initial_clone_index_baf is None:
        x_part, y_part = 3, 3

        initial_clone_index_baf, _ = best_equal_partition(
            coords,
            x_part,
            y_part,
            single_tumor_prop=None,
            threshold=0.5,
        )

        # NB potential initialization strategies, common initial_clone_index_baf, clone_id return:
        # 
        #    initial_clone_index_baf, clone_id = rectangle_initialize_initial_clone(
        #       coords, config.hmrf.n_clones, random_state=0
        #    )
        #
        # initial_clone_index_baf, _ = fixed_rectangle_partition(
        #     coords,
        #     x_part,
        #     y_part,
        #     single_tumor_prop=None,
        #     threshold=0.5,  # random_state=int(config.hmrf.random_state,)
        # )
        #
        # initial_clone_index_baf, _, _ = sufficient_umis_initial_clone(
        #     coords,
        #     single_X[:,0,:],
        #     sample_list,
        #     sample_ids,
        #     500_000, # MAGIC determine by baf.
        #     random_state=int(config.hmrf.random_state),
        # )

    # NB triggers summary for initial clones, per single_X=1, etc; drop return.
    merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index_baf,
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    clone_id = get_clone_assignment(coords, initial_clone_index_baf)
    assignment = pd.Series([f"clone {x}" for x in clone_id])

    initial_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB initial clone assignment for baf-only inference.
    fig_path = f"{plots_dir}/initial_clones_spatial.pdf"
    write_fig(fig_path, initial_clones_fig, transparent=True, bbox_inches="tight")

    pause()

    logger.info(
        "Solving hmm & hmrf for copy states and clone assignment with baf only."
    )

    #
    # ===================================================================
    # baf-derived inference of clone assignment and copy number profiles
    # ===================================================================
    # 

    # NB zero transcript counts for all segments/spots.
    # TODO can drop zero of single_X?  would be useful ...
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    # TODO utilize <BLOCK COUNTS>
    res = hmrfmix_concatenate_pipeline(
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        initial_clone_index_baf,
        config.hmm.n_states,
        log_sitewise_transmat,
        prefix="bafonly",
        coords=coords,
        smooth_mat=smooth_mat,
        adjacency_mat=adjacency_mat,
        sample_ids=sample_ids,
        sample_list=sample_list,
        max_iter_outer=config.hmrf.max_iter_outer,
        hmmclass=hmm_nophasing,
        params="sp",
        t=config.hmm.t,
        random_state=config.hmm.gmm_random_state,
        fix_NB_dispersion=config.hmm.fix_NB_dispersion,
        shared_NB_dispersion=config.hmm.shared_NB_dispersion,
        fix_BB_dispersion=config.hmm.fix_BB_dispersion,
        shared_BB_dispersion=config.hmm.shared_BB_dispersion,
        is_diag=True,
        max_iter=config.hmm.max_iter,
        tol=config.hmm.tol,
        spatial_weight=config.hmrf.spatial_weight,
        tumorprop_threshold=config.hmrf.tumorprop_threshold,
    )

    # NB single_X has dynamic shape (n_segments, 2, n_spots).
    n_obs = single_X.shape[0]

    # NB new pseduo-bulk given new assignment of spots to clones.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(res["new_assignment"], np.unique(res["new_assignment"])),
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(
        f"Inferred {len(np.unique(res['new_assignment']))} clones given baf data."
    )

    # TODO HACK DEPRECATE?
    if tumor_prop is not None:
        tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)

    # TODO HACK
    assignment = pd.Series([f"clone {x}" for x in res["new_assignment"]])
    bafonly_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run.
    fig_path = f"{output_dir}/plots/bafonly_clones_spatial.pdf"
    write_fig(fig_path, bafonly_clones_fig, transparent=True, bbox_inches="tight")

    # TODO copy rename.
    bafonly_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(res["new_assignment"], np.unique(res["new_assignment"])),
        lengths,
        res=res,
        single_tumor_prop=None,
        sample_list=sample_list,
        remove_xticks=True,
    )

    # NB inferred per-clone copy number profiles from baf-only run.
    fig_path = f"{plots_dir}/bafonly_clones_genomic.pdf"
    write_fig(fig_path, bafonly_clones_genomic, transparent=True, bbox_inches="tight")

    pause()

    # NB merge similar clones based on Neyman-Pearson criterion.
    if config.hmrf.np_merge:
        _, merged_res = neyman_pearson_similarity(
            X,
            base_nb_mean,
            total_bb_RD,
            res,
            threshold=config.hmm.np_threshold,
            minlength=config.hmm.np_eventminlen,
            params="sp",
            tumor_prop=tumor_prop,
            hmmclass=hmm_nophasing,
        )
    else:
        logger.warning(f"No Neyman-Pearson merging applied to baf-identified clones.")
        merged_res = res.copy()

    logger.info(
        f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data after NP merge."
    )

    _, merged_res = merge_by_minspots(
        merged_res["new_assignment"],
        merged_res,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=n_obs
        * config.hmrf.min_avgumi_per_clone,  # MAGIC 31_420 SNP UMIs
        single_tumor_prop=single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(
        f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data after min spots merge."
    )

    # TODO HACK
    assignment = pd.Series([f"clone {x}" for x in merged_res["new_assignment"]])
    merged_bafonly_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run, after neyman-pearson "model selection"
    fig_path = f"{output_dir}/plots/merged_bafonly_clones_spatial.pdf"
    write_fig(
        fig_path, merged_bafonly_clones_fig, transparent=True, bbox_inches="tight"
    )

    # TODO copy rename.
    merged_bafonly_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(
            merged_res["new_assignment"], np.unique(merged_res["new_assignment"])
        ),
        lengths,
        res=merged_res,
        single_tumor_prop=None,
        sample_list=sample_list,
        remove_xticks=True,
        rdr_ylim=6,
        chrtext_shift=-0.2,
        base_height=3.2,
        pointsize=5,
        linewidth=1,
    )

    # NB inferred copy number profiles from baf-only run, after neyman-pearson "model selection"
    fig_path = f"{plots_dir}/merged_bafonly_clones_genomic.pdf"
    write_fig(
        fig_path, merged_bafonly_clones_genomic, transparent=True, bbox_inches="tight"
    )

    pause()

    # NB construct data frame with assigned clone label for all samples.
    df_clone_label = pd.DataFrame(
        {
            "sample_id": [barcode.split("_")[-1] for barcode in barcodes],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "clone_label": merged_res["new_assignment"],
        },
        index=barcodes,
    )

    # TODO assert aligned?
    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    # NB cannot sort before barcode-ordered assignments etc!
    df_clone_label = df_clone_label.groupby("sample_id", group_keys=False).apply(
        lambda g: g.sort_values(["x", "y"])
    )

    opath = f"{output_dir}/baf_clone_labels.tsv"

    logger.info(
        f"Writing baf inferred clone labels to {opath},\n{df_clone_label.head()}"
    )

    write_tsv(opath, df_clone_label, header=True, index=True, index_label="barcode")

    # NB single_X has dynamic shape (n_segments, 2, n_spots).
    n_obs = single_X.shape[0]

    # NB clone assignment based on BAF only, after merging similar clones.
    #    number of assigned / selected clones may be less than max. possible ("M").
    merged_baf_assignment = copy.copy(merged_res["new_assignment"])
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # NB predicted copy state (MAP).
    pred = np.argmax(merged_res["log_gamma"], axis=0)

    # NB split into by-clone list vs clone-concatenated array.
    pred = np.array(
        [pred[(c * n_obs) : (c * n_obs + n_obs)] for c in range(n_baf_clones)]
    )

    logger.info(
        f"Found {100. * np.mean(pred[:, :] < config.hmm.n_states)}% of baf-only copy states to have phase 0."
    )

    # DEPRECATE?  baf-only clones are determined with hmm_nophasing.
    # NB contains model baf profiles, accounted for baf-derived phase switching.
    merged_baf_profiles = np.array(
        [
            np.where(
                pred[c, :] < config.hmm.n_states,
                merged_res["new_p_binom"][pred[c, :] % config.hmm.n_states, 0],
                1.0 - merged_res["new_p_binom"][pred[c, :] % config.hmm.n_states, 0],
            )
            for c in range(n_baf_clones)
        ]
    )

    pause()

    # NB normal candidates by baf only (boolean array for n_spots).
    normal_candidate = determine_normal_candidates(
        config,
        merged_res,
        merged_baf_profiles,
        single_X,
        copy_single_X_rdr,
        smooth_mat,
        single_tumor_prop=None,
    )

    index_normal = np.where(normal_candidate)[0]

    pause()

    #
    # ===================================================================
    # clone assignment and copy number profile refinement with UMIs
    # ===================================================================
    # 

    # TODO HACK returns umi information for refinment run with umis.
    single_X[:, 0, :] = copy_single_X_rdr

    # NB filter out genomic segments with potential allele-specific
    #    expression based on normal spot candidates;
    # 
    # TODO normal mis-classification lead to dropped segments due to
    #      identifying CNAs as allele-specific expression.
    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        df_gene_snp,
    ) = normal_baf_bin_filter(
        df_gene_snp,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        config.phasing.nu,
        config.phasing.logphase_shift,
        index_normal,
        config.references.geneticmap_file,
    )

    # NB table of per-bin intervals with set(genes) and set(sites).
    df_bininfo = binned_gene_snp(df_gene_snp)

    # NB update to post-normal filtering single_X.
    copy_single_X_rdr = single_X[:, 0, :]

    # NB filter out high-umi differentially expressed genes, which may
    #    bias RDR estimates.
    if config.quality.filter_normal_diffexp:
        copy_single_X_rdr, _ = filter_normal_diffexp(
            exp_counts,
            df_bininfo,
            normal_candidate,
            sample_list=sample_list,
            sample_ids=sample_ids,
        )
    else:
        logger.warning(f"Assuming no filter for normal differential expression.")

    pause()

    # TODO HACK >>>>>>  do not filter, but merge segments, with insufficient normal umi counts.
    #                   assumes ...  what assumption on phasing, baf-switches?
    df_gene_snp = create_bin_ranges(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        single_X,
        single_total_bb_RD,
        lengths,
        config.quality.secondary_min_umi,
        config.quality.secondary_min_snp_umi,
        config.quality.secondary_min_normal_umi,
        max_binlength=config.quality.max_binlength,
        normal_candidates=normal_candidate,
        key="bin_id",
    )

    df_bininfo = binned_gene_snp(df_gene_snp)

    # TODO separate transmat.
    phase_indicator = np.ones(single_X.shape[0])

    # NB new segmentation and associated counts given normal candidate-based
    #    filtering of baf-derived segments.
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

    copy_single_X_rdr = single_X[:, 0, :]
    # <<<<<<<<<<<<

    # NB >>>>>  zeros single_X_rdr entries with insufficient normal counts,
    #           given config.quality.min_normal_count_perbin.
    _, copy_single_X_rdr, copy_single_base_nb_mean = determine_normal_baseline(
        copy_single_X_rdr,
        normal_candidate,
        config,
    )

    # NB adding back RDR signal
    single_X[:, 0, :] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean

    # NB single_X has dynamic shape (n_segments, 2, n_spots).
    n_obs = single_X.shape[0]
    # <<<<<

    pause()

    exit(0)

    logger.info(
        f"Refinining {n_baf_clones} baf-identified clones with umi data assuming n_clones_rdr={config.hmrf.n_clones_rdr}"
    )

    clone_res = {}

    # NB umi-based refinement of baf-identified clones tries a potentially split only;
    for bafc in range(n_baf_clones):
        logger.info(
            f"-----  Refining baf-identified clone {bafc}/{n_baf_clones}  -----"
        )

        prefix = f"clone{bafc}"

        # NB spots assigned to this baf-only clone (after merging based on Neyman-Pearson similarity).
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        """
        # NB min. b-allele read count (equivalent to 20 per spot) on pseudobulk to split clones.
        if np.sum(single_total_bb_RD[:, idx_spots]) < 20 * single_X.shape[0]:
            logger.warning(
                f"Skipping RDR refinment of baf identified clone {bafc} as too few snp-covering umis ({np.sum(single_total_bb_RD[:, idx_spots]):_}/{20 * single_X.shape[0]:_})!"
            )
            clone_res[prefix] = {
                "barcodes": barcodes[idx_spots],
                "num_iterations": 0,
                "round-1_assignment": np.zeros(len(idx_spots), dtype=int),
                "new_assignment": np.zeros(len(idx_spots), dtype=int),
                "log_gamma": merged_res["log_gamma"][
                    :, (bafc * n_obs) : (bafc * n_obs + n_obs)
                ],  # NB first axis is state.
                "pred_cnv": np.argmax(
                    merged_res["log_gamma"][:, (bafc * n_obs) : (bafc * n_obs + n_obs)],
                    axis=0,
                ),
            }

            continue
        """
        sufficient_snp_umi_for_split = np.sum(single_total_bb_RD[:, idx_spots]) >= 20 * single_X.shape[0]

        # NB initialize new set of clones within this baf identified clone.
        # TODO tumor_prop, i.e. _mix.
        initial_clone_index, _ = rectangle_initialize_initial_clone(
            coords[idx_spots],
            config.hmrf.n_clones_rdr if sufficient_snp_umi_for_split else 1,
            random_state=0,  # TODO HACK.
        )

        # # TODO HACK?  splits each BAF clone along the x direction.
        # # TODO BUG require min spots/umis etc ...
        # x_part, y_part = config.hmrf.n_clones_rdr, 1

        # initial_clone_index, _ = fixed_rectangle_partition(
        #     coords[idx_spots],
        #     x_part,
        #     y_part,
        # )

        initial_assignment = np.zeros(len(idx_spots), dtype=int)

        # NB zero-indexes clones.
        for c, idx in enumerate(initial_clone_index):
            initial_assignment[idx] = c

        # NB barcodes contained within this BAF-identified clone.
        clone_res[prefix] = {
            "barcodes": barcodes[idx_spots],
            "num_iterations": 0,
            "round-1_assignment": initial_assignment,
        }

        # NB slice ids for each spot in this clone.
        copy_slice_sample_ids = copy.copy(sample_ids[idx_spots])

        # TODO HACK
        copy_slice_sample_list = list(
            np.unique(np.array(sample_list)[sample_ids[idx_spots]])
        )

        # NB hmrf + hmm with RDR data.
        new_clone_res = hmrfmix_concatenate_pipeline(
            single_X[:, :, idx_spots],
            lengths,
            single_base_nb_mean[
                :, idx_spots
            ],  # NB per-spot replications normalized to T_n.
            single_total_bb_RD[:, idx_spots],
            single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
            initial_clone_index,
            n_states=config.hmm.n_states,
            prefix=prefix,
            coords=coords[idx_spots],
            log_sitewise_transmat=log_sitewise_transmat,
            smooth_mat=smooth_mat[np.ix_(idx_spots, idx_spots)],
            adjacency_mat=adjacency_mat[np.ix_(idx_spots, idx_spots)],
            sample_ids=copy_slice_sample_ids,
            sample_list=copy_slice_sample_list,
            max_iter_outer=config.hmrf.max_iter_outer,
            hmmclass=hmm_nophasing,
            params="smp",
            t=config.hmm.t,
            random_state=config.hmm.gmm_random_state,
            fix_NB_dispersion=config.hmm.fix_NB_dispersion,
            shared_NB_dispersion=config.hmm.shared_NB_dispersion,
            fix_BB_dispersion=config.hmm.fix_BB_dispersion,
            shared_BB_dispersion=config.hmm.shared_BB_dispersion,
            is_diag=True,
            max_iter=config.hmm.max_iter,
            tol=config.hmm.tol,
            spatial_weight=config.hmrf.spatial_weight,
            tumorprop_threshold=config.hmrf.tumorprop_threshold,
            init_p_binom=None,
            init_log_mu=None,
        )

        clone_res[prefix] = merge_dicts(clone_res[prefix], new_clone_res)

        pause()

    logger.info(f"Found initial solutions for baf-identified clones refined by rdr.")
    logger.info(f"Combining results across clones.")

    # NB combined assignment for all spots.
    res_combine = {"prev_assignment": np.zeros(single_X.shape[2], dtype=int)}
    offset_clone = 0

    # NB Neyman-Pearson and min. spot merging across baf clones split by rdr.
    for bafc in range(n_baf_clones):
        prefix = f"clone{bafc}"
        res = clone_res[prefix]

        idx_spots = np.where(barcodes.isin(res["barcodes"]))[0]

        # NB baf clone was not split.
        if len(np.unique(res["new_assignment"])) == 1:
            logger.info(f"clone {bafc} was not split by rdr.")

            # NB clone id.
            c, n_merged_clones = res["new_assignment"][0], 1

            # NB merging is a null op.
            merged_res = copy.copy(res)
            merged_res["new_assignment"] = np.zeros(len(idx_spots), dtype=int)

            # NB c must be zero here (1 clone, zero-indexed).
            log_gamma = res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)].reshape(
                (-1, n_obs, 1)
            )

            # NB MAP copy state - both this and log_gamma should be null-ops as already correct shape.
            pred_cnv = res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)].reshape(
                (-1, 1)
            )
        else:
            clone_index = [
                np.where(res["new_assignment"] == c)[0]
                for c in np.sort(np.unique(res["new_assignment"]))
            ]

            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
                single_X[:, :, idx_spots],
                single_base_nb_mean[:, idx_spots],
                single_total_bb_RD[:, idx_spots],
                clone_index,
                single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
                threshold=config.hmrf.tumorprop_threshold,
            )

            if tumor_prop is not None:
                tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)

            if config.hmrf.np_merge:
                # NB merge rdr split clones (within baf clone) based on Neyman-Pearson similarity;
                #    does not account for similarity across baf-clones.
                _, merged_res = neyman_pearson_similarity(
                    X,
                    base_nb_mean,
                    total_bb_RD,
                    res,
                    threshold=config.hmm.np_threshold,
                    minlength=config.hmm.np_eventminlen,
                    params="smp",
                    tumor_prop=tumor_prop,
                    hmmclass=hmm_nophasing,
                )
            else:
                logger.warning(
                    "No Neyman-Pearson merging applied to RDR identified clones."
                )
                merged_res = res.copy()

            merging_groups, merged_res = merge_by_minspots(
                merged_res["new_assignment"],
                merged_res,
                single_total_bb_RD[:, idx_spots],
                min_spots_thresholds=config.hmrf.min_spots_per_clone,
                min_umicount_thresholds=n_obs
                * config.hmrf.min_avgumi_per_clone,  # MAGIC 31_420 SNP UMIs
                single_tumor_prop=(
                    single_tumor_prop[idx_spots]
                    if single_tumor_prop is not None
                    else None
                ),
                threshold=config.hmrf.tumorprop_threshold,
            )

            # NB num. of rdr-split clones within baf clone after merging.
            n_merged_clones = len(merging_groups)
            fixed_assignment = copy.copy(merged_res["new_assignment"])

            # NB compute posterior using the newly merged pseudobulk
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
                single_X[:, :, idx_spots],
                single_base_nb_mean[:, idx_spots],
                single_total_bb_RD[:, idx_spots],
                get_clone_indices(
                    merged_res["new_assignment"], range(n_merged_clones)
                ),  # TODO clone_ids def. vs range
                single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
                threshold=config.hmrf.tumorprop_threshold,
            )

            # NB recompute copy states and clone profiles based on new pseudobulk.  As a result,
            #    (rdr, baf) copy states per clone vs universal.
            # TODO clone stack.
            merged_res = pipeline_baum_welch(
                None,
                np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
                    -1, 2, 1
                ),
                np.tile(lengths, X.shape[2]),
                config.hmm.n_states,
                base_nb_mean.flatten("F").reshape(-1, 1),
                total_bb_RD.flatten("F").reshape(-1, 1),
                np.tile(log_sitewise_transmat, X.shape[2]),
                (
                    np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
                    if not tumor_prop is None
                    else None
                ),
                hmmclass=hmm_nophasing,
                params="smp",
                t=config.hmm.t,
                random_state=config.hmm.gmm_random_state,
                fix_NB_dispersion=config.hmm.fix_NB_dispersion,
                shared_NB_dispersion=config.hmm.shared_NB_dispersion,
                fix_BB_dispersion=config.hmm.fix_BB_dispersion,
                shared_BB_dispersion=config.hmm.shared_BB_dispersion,
                is_diag=True,
                init_log_mu=res["new_log_mu"],
                init_p_binom=res["new_p_binom"],
                init_alphas=res["new_alphas"],
                init_taus=res["new_taus"],
                max_iter=config.hmm.max_iter,
                tol=config.hmm.tol,
                lambd=np.sum(base_nb_mean, axis=1) / np.sum(base_nb_mean),
                sample_length=np.ones(X.shape[2], dtype=int) * X.shape[0],
            )

            # NB assignment has been fixed, but emission states updated; retain previous assignment.
            merged_res["new_assignment"] = copy.copy(fixed_assignment)

            # NB combined only between similar states in the RDR split clones by updating res["pred_cnv"]
            merged_res = combine_similar_states_across_clones(
                X,
                base_nb_mean,
                total_bb_RD,
                merged_res,
                params="smp",
                tumor_prop=(
                    np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
                    if not tumor_prop is None
                    else None
                ),
                hmmclass=hmm_nophasing,
                merge_threshold=config.hmm.np_merge_threshold,  # MAGIC 0.1
            )

            log_gamma = np.stack(
                [
                    merged_res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)]
                    for c in range(n_merged_clones)
                ],
                axis=-1,
            )
            pred_cnv = np.vstack(
                [
                    merged_res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)]
                    for c in range(n_merged_clones)
                ]
            ).T

        # NB res_combine has the "prev_assignment" key only.
        if len(res_combine) == 1:
            res_combine.update(
                {
                    "new_log_mu": np.hstack(
                        n_merged_clones * [merged_res["new_log_mu"]]
                    ),
                    "new_alphas": np.hstack(
                        n_merged_clones * [merged_res["new_alphas"]]
                    ),
                    "new_p_binom": np.hstack(
                        n_merged_clones * [merged_res["new_p_binom"]]
                    ),
                    "new_taus": np.hstack(n_merged_clones * [merged_res["new_taus"]]),
                    "log_gamma": log_gamma,
                    "pred_cnv": pred_cnv,
                }
            )
        else:
            res_combine.update(
                {
                    "new_log_mu": np.hstack(
                        [res_combine["new_log_mu"]]
                        + n_merged_clones * [merged_res["new_log_mu"]]
                    ),
                    "new_alphas": np.hstack(
                        [res_combine["new_alphas"]]
                        + n_merged_clones * [merged_res["new_alphas"]]
                    ),
                    "new_p_binom": np.hstack(
                        [res_combine["new_p_binom"]]
                        + n_merged_clones * [merged_res["new_p_binom"]]
                    ),
                    "new_taus": np.hstack(
                        [res_combine["new_taus"]]
                        + n_merged_clones * [merged_res["new_taus"]]
                    ),
                    "log_gamma": np.dstack([res_combine["log_gamma"], log_gamma]),
                    "pred_cnv": np.hstack([res_combine["pred_cnv"], pred_cnv]),
                }
            )

        res_combine["prev_assignment"][idx_spots] = (
            merged_res["new_assignment"] + offset_clone
        )

        logger.info(
            f"baf-identified clone={bafc} generated rdr-split clones={np.unique(merged_res['new_assignment'] + offset_clone)}"
        )

        offset_clone += n_merged_clones

        pause()

    # TODO BUG?? prev_assignment or new_assignment?
    n_final_clones = len(np.unique(res_combine["prev_assignment"]))

    logger.info(f"Inferred {n_final_clones} clones given rdr & baf data.")

    logger.info(f"Found rdr-split clone rdrs:\n{np.exp(res_combine['new_log_mu'])}.")
    logger.info(f"Found rdr-split clone bafs:\n{res_combine['new_p_binom']}.")

    logger.info(
        f"Assuming max. alpha dispersion={np.max(res_combine['new_alphas']):.4f} between clones given current:\n{res_combine['new_alphas']}"
    )
    logger.info(
        f"Assuming min. tau dispersion={np.min(res_combine['new_taus']):.4f} between clones given current:\n{res_combine['new_taus']}"
    )

    # HACK broadcast max. dispersion - parameters assumed to be shared across rdr-split clones only.
    res_combine["new_alphas"][:, :] = np.max(res_combine["new_alphas"])

    # HACK broadcast min. dispersion across all clones; tau is total pseduocounts for BAF
    #      min. is least significant.
    res_combine["new_taus"][:, :] = np.min(res_combine["new_taus"])

    pause()

    log_persample_weights = np.zeros((n_final_clones, len(sample_list)))

    for sidx in range(len(sample_list)):
        index = np.where(sample_ids == sidx)[0]
        this_persample_weight = np.bincount(
            res_combine["prev_assignment"][index], minlength=n_final_clones
        ) / len(index)
        log_persample_weights[:, sidx] = np.where(
            this_persample_weight > 0, np.log(this_persample_weight), -50
        )
        log_persample_weights[:, sidx] = log_persample_weights[
            :, sidx
        ] - scipy.special.logsumexp(log_persample_weights[:, sidx])

    pred = np.vstack(
        [
            np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
            for c in range(res_combine["log_gamma"].shape[2])
        ]
    ).T

    # NB final re-assignment across all spots using current copy states -
    #    does not conserve original e.g. baf clone assignments, or normal spots.
    #    Further, does not assume same clone concatenated shape.
    logger.info(f"Finalizing clone assignment with refined parameters.")

    if config.preprocessing.tumorprop_file is None:
        new_assignment, _, total_llf, _ = aggr_hmrf_reassignment(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res_combine,
            pred,
            smooth_mat,
            adjacency_mat,
            res_combine["prev_assignment"],
            copy.copy(sample_ids),
            log_persample_weights,
            spatial_weight=config.hmrf.spatial_weight,
            hmmclass=hmm_nophasing,
            return_posterior=True,
        )
    else:
        (
            new_assignment,
            _,
            total_llf,
            _,
        ) = aggr_hmrfmix_reassignment(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            res_combine,
            pred,
            smooth_mat,
            adjacency_mat,
            res_combine["prev_assignment"],
            copy.copy(sample_ids),
            log_persample_weights,
            spatial_weight=config.hmrf.spatial_weight,
            hmmclass=hmm_nophasing,
            return_posterior=True,
        )

    # NB total Potts likelihood given final copy states and clone assignment.
    res_combine["total_llf"] = total_llf
    res_combine["new_assignment"] = new_assignment

    # NB re-order clones such that the index of the most-normal clone is 0.
    res_combine, _ = reindex_clones(res_combine, posterior=None, single_tumor_prop=None)

    final_clones, final_clone_counts = np.unique(
        res_combine["new_assignment"], return_counts=True
    )

    logger.info(
        f"Inferred final clones=\n{final_clones}\nwith fractions=\n{final_clone_counts/np.sum(final_clone_counts)}."
    )

    pause()

    # TODO new_log_startprob - add to res_combine above.
    for key in [
        "new_log_mu",
        "new_alphas",
        "new_p_binom",
        "new_taus",
        "total_llf",
        "pred_cnv",
    ]:
        logger.info(f"Solved for {key}:\n{res_combine[key]}")

    # TODO SIC BUG params?
    np.savez(
        f"{output_dir}/rdrbaf_final_nstates{config.hmm.n_states}_smp.npz", **res_combine
    )

    pause()

    # NB infer integer allele-specific copy numbers
    final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))

    # DEPRECATE
    # NB add normal clone as 0 if not present
    if 0 not in final_clone_ids:
        final_clone_ids = np.append(0, final_clone_ids)
        logger.error(f"Missing normal clones - prepended as 0 to final clone ids.")
        raise RuntimeError()

    logger.info(f"Utilizing final clone ids={final_clone_ids}")

    pause()

    # NB assumed ploidy for integer copy number problem
    medfix = [""] + [f"_{pp}" for pp in config.int_copy_num.ploidy.split(",")]

    int_ploidy_map = {"": None, "diploid": 2, "triploid": 3, "tetraploid": 4}
    int_ploidy = [int_ploidy_map[key.replace("_", "")] for key in medfix]

    for o, max_medploidy in enumerate(int_ploidy):
        logger.info(
            f"Solving integer copy number problem for max_medploidy={max_medploidy}."
        )

        # NB A/B integer copy number per bin and per state
        allele_specific_copy, state_cnv = [], []
        df_genelevel_cnv = None

        # NB pseudobulk for each of the final clones.
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            [
                np.where(res_combine["new_assignment"] == cid)[0]
                for cid in final_clone_ids
            ],
            single_tumor_prop,
            threshold=config.hmrf.tumorprop_threshold,
        )

        for s, cid in enumerate(final_clone_ids):
            if np.sum(base_nb_mean[:, s]) == 0:
                logger.warning("Final clone {cid} has no assigned transcripts.")
                continue

            this_pred_cnv = res_combine["pred_cnv"][:, s]

            # NB log state usage
            us, cnts = np.unique(this_pred_cnv, return_counts=True)

            # TODO HACK
            logger.info(
                f"Found state usage for clone {cid}:\n{pd.DataFrame({'state': us, 'counts': cnts})}"
            )

            # NB adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1.
            lambd = base_nb_mean[:, s] / np.sum(base_nb_mean[:, s])
            adjusted_log_mu = (
                np.log(
                    np.exp(res_combine["new_log_mu"][:, s])
                    / np.sum(
                        np.exp(res_combine["new_log_mu"][this_pred_cnv, s]) * lambd
                    )
                )
                if config.run.legacy
                else res_combine["new_log_mu"][:, s]
            )  # TODO HACK BUG?

            logger.info(
                f"For clone {cid}, normalized log mu to sum_bin lambda * np.exp(log_mu) = 1.; yielding new mu=\n{np.exp(adjusted_log_mu)}\ngiven mu=\n{np.exp(res_combine["new_log_mu"][:, s])}."
            )

            # TODO finalize integer copy number determination.
            if max_medploidy is not None:
                best_integer_copies, loss = hill_climbing_integer_copynumber_oneclone(
                    adjusted_log_mu,
                    base_nb_mean[:, s],
                    res_combine["new_p_binom"][:, s],
                    this_pred_cnv,
                    max_medploidy=max_medploidy,
                )
            else:
                (
                    best_integer_copies,
                    loss,
                ) = hill_climbing_integer_copynumber_fixdiploid(
                    adjusted_log_mu,
                    base_nb_mean[:, s],
                    res_combine["new_p_binom"][:, s],
                    this_pred_cnv,
                    nonbalance_bafdist=config.int_copy_num.nonbalance_bafdist,
                    nondiploid_rdrdist=config.int_copy_num.nondiploid_rdrdist,
                    # min_prop_threshold=0.02,  # MAGIC
                )

                # TODO HACK
                # finding_distate_failed = True
                # continue

            logger.info(
                f"Solved for (max. med ploidy, clone) = ({max_medploidy}, {s}) with integer copy number loss = {loss:.4e}"
            )

            # NB best integer copies for each clone and each ploidy.
            allele_specific_copy.append(
                pd.DataFrame(
                    this_pred_cnv.reshape(1, -1),
                    index=[f"clone{cid} Z"],
                    columns=np.arange(n_obs),
                )
            )

            allele_specific_copy.append(
                pd.DataFrame(
                    res_combine["new_log_mu"][this_pred_cnv, s].reshape(1, -1),
                    index=[f"clone{cid} logmu"],
                    columns=np.arange(n_obs),
                )
            )

            allele_specific_copy.append(
                pd.DataFrame(
                    res_combine["new_p_binom"][this_pred_cnv, s].reshape(1, -1),
                    index=[f"clone{cid} p"],
                    columns=np.arange(n_obs),
                )
            )

            allele_specific_copy.append(
                pd.DataFrame(
                    best_integer_copies[this_pred_cnv, 0].reshape(1, -1),
                    index=[f"clone{cid} A"],
                    columns=np.arange(n_obs),
                )
            )
            allele_specific_copy.append(
                pd.DataFrame(
                    best_integer_copies[this_pred_cnv, 1].reshape(1, -1),
                    index=[f"clone{cid} B"],
                    columns=np.arange(n_obs),
                )
            )

            state_cnv.append(
                pd.DataFrame(
                    res_combine["new_log_mu"][:, s].reshape(-1, 1),
                    columns=[f"clone{cid} logmu"],
                    index=np.arange(config.hmm.n_states),
                )
            )
            state_cnv.append(
                pd.DataFrame(
                    res_combine["new_p_binom"][:, s].reshape(-1, 1),
                    columns=[f"clone{cid} p"],
                    index=np.arange(config.hmm.n_states),
                )
            )
            state_cnv.append(
                pd.DataFrame(
                    best_integer_copies[:, 0].reshape(-1, 1),
                    columns=[f"clone{cid} A"],
                    index=np.arange(config.hmm.n_states),
                )
            )
            state_cnv.append(
                pd.DataFrame(
                    best_integer_copies[:, 1].reshape(-1, 1),
                    columns=[f"clone{cid} B"],
                    index=np.arange(config.hmm.n_states),
                )
            )

            bin_Acopy_mappers = {
                i: x
                for i, x in enumerate(
                    best_integer_copies[res_combine["pred_cnv"][:, s], 0]
                )
            }
            bin_Bcopy_mappers = {
                i: x
                for i, x in enumerate(
                    best_integer_copies[res_combine["pred_cnv"][:, s], 1]
                )
            }
            tmpdf = pd.DataFrame(
                {
                    "gene": df_gene_snp[df_gene_snp.is_interval].gene,
                    f"clone{s} A": df_gene_snp[df_gene_snp.is_interval]["bin_id"].map(
                        bin_Acopy_mappers
                    ),
                    f"clone{s} B": df_gene_snp[df_gene_snp.is_interval]["bin_id"].map(
                        bin_Bcopy_mappers
                    ),
                }
            ).set_index("gene")

            if df_genelevel_cnv is None:
                df_genelevel_cnv = copy.copy(
                    tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int)
                )
            else:
                df_genelevel_cnv = df_genelevel_cnv.join(
                    tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int)
                )

            pause()

        if len(state_cnv) == 0:
            logger.warning(f"Found empty state integer copy numbers for clone{s}!")
            continue

        # logger.info(
        #     f"Solved for integer copy numbers @ genes:\n{df_genelevel_cnv.head()}"
        # )

        opath = f"{output_dir}/cnv{medfix[o]}_genelevel.tsv"

        # NB output gene-level copy number
        write_tsv(opath, df_genelevel_cnv, header=True, index=True)

        # NB output segment-level copy number
        allele_specific_copy = pd.concat(allele_specific_copy)
        df_seglevel_cnv = pd.DataFrame(
            {
                "CHR": df_bininfo.CHR.values,
                "START": df_bininfo.START.values,
                "END": df_bininfo.END.values,
            }
        )
        df_seglevel_cnv = df_seglevel_cnv.join(allele_specific_copy.T)

        a_cols = [c for c in df_seglevel_cnv.columns if c.endswith(" A")]
        b_cols = [c.replace(" A", " B") for c in a_cols]
        mask = (df_seglevel_cnv[a_cols].ne(1) | df_seglevel_cnv[b_cols].ne(1)).any(
            axis=1
        )

        with pd.option_context(
            "display.expand_frame_repr",
            False,
            "display.max_columns",
            None,
            "display.width",
            100000,
            "display.max_colwidth",
            None,
        ):
            logger.info(
                "Solved for integer copy numbers @ segments:\n%s",
                df_seglevel_cnv[mask].to_string(index=False),
            )

        opath = f"{output_dir}/cnv{medfix[o]}_seglevel.tsv"
        write_tsv(opath, df_seglevel_cnv, header=True, index=False)

        # NB output per-state copy number
        state_cnv = functools.reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="inner"
            ),
            state_cnv,
        )

        with pd.option_context(
            "display.expand_frame_repr",
            False,
            "display.max_columns",
            None,
            "display.width",
            None,
            "display.max_colwidth",
            None,
        ):
            logger.info(
                "Solved for integer copy numbers @ states:\n%s",
                state_cnv.to_string(index=False),
            )

        opath = f"{output_dir}/cnv{medfix[o]}_perstate.tsv"
        write_tsv(opath, state_cnv, header=True, index=False)

        copy_states_fig = plot_copy_states(state_cnv)
        write_fig(
            f"{plots_dir}/copy_states{medfix[o]}.pdf",
            copy_states_fig,
            transparent=True,
            bbox_inches="tight",
        )

    df_clone_label = pd.DataFrame(
        {
            "sample_id": [barcode.split("_")[-1] for barcode in barcodes],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "clone_label": res_combine["new_assignment"],
        },
        index=barcodes,
    )

    # TODO assert aligned?
    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    # NB cannot sort before barcode-ordered assignments etc!
    df_clone_label = df_clone_label.groupby("sample_id", group_keys=False).apply(
        lambda g: g.sort_values(["x", "y"])
    )

    opath = f"{output_dir}/clone_labels.tsv"

    logger.info(f"Writing inferred clone labels to {opath},\n{df_clone_label.head()}")

    write_tsv(opath, df_clone_label, header=True, index=True, index_label="barcode")

    rdr_baf_fig = plot_clones_genomic(
        df_seglevel_cnv,
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        res_combine,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        clone_ids=None,
        remove_xticks=True,
        chrtext_shift=-0.3,
        base_height=3.2,
        palette_name="chisel",
    )

    # TODO
    fig_path = f"{output_dir}/plots/clones_genomic.pdf"
    write_fig(fig_path, rdr_baf_fig, transparent=True, bbox_inches="tight")

    # TODO issue when indexing of initial clones incompatiable/bigger than final clones.
    # initial_rdr_baf_fig = plot_clones_genomic(
    #     df_seglevel_cnv,
    #     lengths,
    #     single_X,
    #     single_base_nb_mean,
    #     single_total_bb_RD,
    #     res_combine,
    #     single_tumor_prop=single_tumor_prop,
    #     sample_list=sample_list,
    #     clone_ids=None,
    #     clone_index=initial_clone_index_baf,
    #     remove_xticks=True,
    #     base_height=3.2,
    #     palette_name="chisel",
    # )

    # # TODO
    # fig_path = f"{output_dir}/plots/initial_clones_genomic.pdf"
    # write_fig(fig_path, initial_rdr_baf_fig, transparent=True, bbox_inches="tight")

    clone_index = [
        np.where(res_combine["new_assignment"] == c)[0]
        for c, _ in enumerate(final_clone_ids)
    ]

    # NB create pseudobulk for each clone.
    X, base_nb_mean, total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    # NB clones fig.
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    fig_path = f"{output_dir}/plots/clones_spatial.pdf"
    write_fig(fig_path, clones_fig, transparent=True, bbox_inches="tight")

    clone_index = [
        np.where(res_combine["new_assignment"] == c)[0]
        for c, _ in enumerate(final_clone_ids)
    ]
    # NB create pseudobulk for each clone.
    X, base_nb_mean, total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )
    """
    plot_cna_mixture(
        res_combine["new_log_mu"],
        res_combine["new_alphas"],
        res_combine["new_p_binom"],
        res_combine["new_taus"],
        X,
        base_nb_mean,
        total_bb_RD,
        prefix="final",
    )
    """
    logger.info(f"Done in {(time.time() - start_time)/60.:.2f} minutes.")


# NB run_cnaster zenodo_sim_config (-o paths.sample_sheet='dummy_sample_sheet.tsv')
def main():
    parser = argparse.ArgumentParser(description="Run CNAster pipeline")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--over_rides",
        "-o",
        action="append",
        default=[],
        help="Over ride config keys in dot notation, e.g. -o paths.sample_sheet=/path/to/sheet.csv",
    )

    args = parser.parse_args()

    run_cnaster(args.config_path, over_rides=args.over_rides)


if __name__ == "__main__":
    main()
