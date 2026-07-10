import argparse
import copy
import functools
import random
import time

import numpy as np
import pandas as pd
from numba import njit

from cnaster.config import YAMLConfig, set_global_config, start_time
from cnaster.he import get_he_image
from cnaster.hmm_nophasing import hmm_nophasing, get_log_transmat
from cnaster.sandbox.hmm_nophasing_jax_v2 import hmm_nophasing_jax

from cnaster.hmrf import merge_by_minspots, reindex_clones, run_core_inference
from cnaster.hmrf_utils import get_clone_assignment, get_clone_indices
from cnaster.integer_copy import (
    hill_climbing_integer_copynumber_fixdiploid_milp,
    hill_climbing_integer_copynumber_oneclone,
)
from cnaster.io import (
    construct_df_clone_label,
    get_sample_list,
    load_input_data,
    read_tumor_prop,
)
from cnaster.logger import get_logger
from cnaster.normal_spot import (
    determine_normal_baseline,
    determine_normal_candidates,
    filter_normal_diffexp,
    normal_baf_bin_filter,
)
from cnaster.omics import (
    assign_initial_blocks,
    create_bin_ranges,
    form_gene_snp_table,
    binned_gene_snp,
    get_sitewise_transmat,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnaster.phasing import initial_phase_given_partition
from cnaster.plot_copy_number_profile import plot_copy_number_profile
from cnaster.plot_genomic import plot_clones_genomic
from cnaster.plotting import plot_clones_spatial, plot_he
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.spatial import (
    best_equal_partition,
    initialize_clones,
    initialize_rdr_clone_refininement,
    construct_multislice_lattice_adjacency,
)
from cnaster.utils import configure_output_dir, pause, write_fig, write_tsv
from cnaster.annotation import load_clone_labels, load_clone_ranges, assign_clone_ranges

# from cnaster.sim import load_tables_to_matrices
# from cnaster.hmm_phased import hmm_phased
# from cnaster.plot_loh_density import plot_loh_density
# from cnaster.adjacency import multislice_adjacency as multislice_adjacency_simple
# from cnaster.wolff import initialize_clones_wolff


@njit
def set_numba_seed(value):
    np.random.seed(value)


logger = get_logger(__name__, start_time=start_time)


def run_cnaster(config_path, over_rides=None):
    logger.runtime_phase = "prep."
    logger.info("----  Welcome to cna-maste  ----")

    config = YAMLConfig.from_file(config_path)
    config.over_ride(over_rides)
    config.issue_warnings()

    logger.info(f"Read configuration:\n{config}")

    set_global_config(config)

    output_dir, plots_dir = configure_output_dir(config)

    logger.info(f"Set (numpy) random seed={config.hmrf.random_state}")

    # TODO fix reproducibility - set random seed globally.
    np.random.seed(int(config.hmrf.random_state))
    random.seed(int(config.hmrf.random_state))
    set_numba_seed(int(config.hmrf.random_state))

    # NB legacy simulated data loading - generates matrices for all steps of the pipeline.
    # (
    #     lengths,
    #     single_X,
    #     single_base_nb_mean,
    #     single_total_bb_RD,
    #     log_sitewise_transmat,
    #     df_bin_info,
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

    # TODO move to load_input_data
    #
    # NB sample list derived from adata.obs['sample'] - removes adjacent duplicates.
    #    sample_ids: unique enum for each entry in sample_list.  One per adata.obs entry.
    sample_list, sample_ids = get_sample_list(adata)
    single_tumor_prop = read_tumor_prop(adata, config=config)
    """
    # TODO HACK
    adjacency_mat, smooth_mat = multislice_adjacency_simple(
        coords, 
        sample_ids, 
        lattice_type=None,
        n_nearest=6,
        dx=1.0, 
        dy=1.0, # sqrt(3)/2 for hexagonal lattice
    )
    """
    # NB parse_visium::combine_gene_snps
    #    [ chr, start, end, snp_id, gene, is_interval (is_gene) ]
    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )

    if config.annotation.clone_ranges is not None:
        clone_ranges = load_clone_ranges(config.annotation.clone_ranges)
        df_gene_snp = assign_clone_ranges(df_gene_snp, clone_ranges, key="known_id", collapse=False)

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
    # NB lengths = num. of blocks per contig;
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
        "block_id",
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    # NB pseudobulk formed of all spots.
    initial_clone_pseudobulk = [[ii for ii in range(len(coords))]]

    pseudobulk_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=initial_clone_pseudobulk,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
    )

    write_fig(
        f"{plots_dir}/pseudobulk_clones_genomic.pdf",
        pseudobulk_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    #
    # ============================================================
    # baf-derived phasing (assuming initial / h&e derived clones
    # ============================================================
    #
    logger.runtime_phase = "phasing"

    # TODO
    known_clone_assignment = known_nb_baseline = None

    if config.annotation.clone_label is not None:
        known_clone_assignment, known_nb_baseline = load_clone_labels(
            single_X, config
        )
        initial_clone_for_phasing = known_clone_assignment.copy()

    else:
        # NB  rectangular partition across multiple slices, equivalent to parse_visium::perform_partition.
        initial_clone_for_phasing = initialize_clones(
            coords,
            sample_ids,  # NB for all spots in all slices.
            x_part=config.phasing.npart_phasing,
            y_part=config.phasing.npart_phasing,
            config=config,
        )
        """
        initial_clone_for_phasing = initialize_clones_wolff(
            sample_ids,
            adjacency_mat,
            n_init=1,
            base_n_clones=5,
            spatial_weight=0.65,
            wolff_num_temps=1,
            wolff_sweeps_per_temp=1_000,
            min_spots=100,
            random_state=None,
            config=None,
            relabel=True,
            min_temp=1.,
            max_temp=1.,
        ).pop()
        """

    # TODO
    initial_clone_index_baf = initial_clone_for_phasing

    # NB utilize initial spot assignment based on h&e image; potts model may (will!) merge, or blur h&e boundaries.
    if "he_label" in adata.obsm:
        logger.info(f"Refining initial clone partition with h&e derived segmentation.")

        spatial_assignment = get_clone_assignment(coords, initial_clone_for_phasing)

        # NB per-spot h&e label derived from gray-scale percentiles.
        he_assignment = adata.obsm["he_label"].flatten()

        # TODO FutureWarning: factorize with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.z
        # TODO comments;
        clone_assignment, _ = pd.factorize(list(zip(he_assignment, spatial_assignment)))

        initial_clone_for_phasing = initial_clone_index_baf = get_clone_indices(
            clone_assignment,
            np.unique(clone_assignment),
        )

        # TODO constructor:  defined for all spots, in_tissue, etc?
        he_frame = get_he_image(
            spaceranger_dir=config.preprocessing.spaceranger_dir,
            res="hires",
            pos=None,
            num_labels=4,
        )

        he_fig = plot_he(
            he_frame,
            channels=("image"),
        )

        write_fig(
            f"{plots_dir}/he_image.pdf",
            he_fig,
            transparent=True,
            bbox_inches="tight",
        )

    assignment = pd.Series(
        [f"clone {x}" for x in get_clone_assignment(coords, initial_clone_for_phasing)]
    )

    phasing_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    # NB plot of the clones assumed for initial phasing.
    write_fig(
        f"{plots_dir}/prephasing_clones_spatial.pdf",
        phasing_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    prephasing_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=initial_clone_for_phasing,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        known_nb_baseline=known_nb_baseline,
    )

    write_fig(
        f"{plots_dir}/prephasing_clones_genomic.pdf",
        prephasing_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    exit(0)

    if config.phasing.run:
        n_states_phasing = config.hmm.n_states
        log_transmat = get_log_transmat(config.hmm.n_states, config.hmm.t)

        # NB single_base_nb_mean initialized to zero - requires normal spot determination.
        res_phasing, phase_indicator, refined_lengths = initial_phase_given_partition(
            single_X,
            lengths,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            initial_clone_for_phasing,
            n_states_phasing,
            log_transmat,
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

        # TODO PATCH
        res_phasing["new_assignment"] = get_clone_assignment(
            coords, initial_clone_for_phasing
        )

        logger.info(
            f"Solution for initial phase given pop. phasing (eagle) & observed baf in {(time.time() - start_time):.2f} seconds."
        )
    else:
        # TODO comment
        phase_indicator = np.zeros(single_X.shape[0])
        refined_lengths = lengths

    # NB phase is None for genes and otherwise 0/1 for snps given baf-inferred phase.
    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

    postphasing_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=initial_clone_for_phasing,
        res_combine=res_phasing,
        single_tumor_prop=None,
        sample_list=sample_list,
    )

    write_fig(
        f"{plots_dir}/postphasing_clones_genomic.pdf",
        postphasing_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    logger.runtime_phase = "phased genomic segmentation"

    # NB generates new genomic intervals ("bin_id") by genomic aggregation
    #    accounting for baf-derived phasing and user defined thresholds.
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

    logger.info(
        f"Recalculating counts for new (phased) baf inferred genome segmentation."
    )

    # TODO summarize_counts_for_blocks can be adapted to summarize_counts_for_bins,
    #      given new df_gene_snp with "bin_id" and "phase" columns.
    #
    # NB   counters per baf-phasing derived genomic intervals.
    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
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

    log_sitewise_transmat = get_sitewise_transmat(
        segment_key="bin_id",
        df_gene_snp=df_gene_snp,
        geneticmap_file=config.references.geneticmap_file,
        nu=config.phasing.nu,
        logphase_shift=config.phasing.logphase_shift,
    )

    pause()

    postphasing_aggr_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=initial_clone_for_phasing,
        res_combine=None,
        single_tumor_prop=None,
        sample_list=sample_list,
    )

    write_fig(
        f"{plots_dir}/postphasing_aggr_clones_genomic.pdf",
        postphasing_aggr_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pseudobulk_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=initial_clone_pseudobulk,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
    )

    write_fig(
        f"{plots_dir}/postphasing_pseudobulk_clones_genomic.pdf",
        pseudobulk_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    #
    # ===================================================================
    # baf-derived inference of clone assignment and copy number profiles
    # ===================================================================
    #

    logger.runtime_phase = "baf-only clone & copy state inference"

    # TODO
    # NB smooth pooling matrix & distance based (exponential decay) adjacency.
    #    requires pre-defined single_total_bb_RD, but largely on data loading.
    adjacency_mat, smooth_mat = construct_multislice_lattice_adjacency(
        sample_ids,
        sample_list,
        coords,
        across_slice_adjacency_mat,
        maxspots_pooling=1,  # DEPRECATE config.hmrf.maxspots_pooling,
        unit_xsquared=config.hmrf.unit_xsquared,  # TODO
        unit_ysquared=config.hmrf.unit_ysquared,  # TODO
    )

    # adjacency_fig = plot_adjacency(
    #     coords,
    #     smooth_mat,
    #     adjacency_mat,
    #     pointsize=5,
    #     base_height=6,
    #     sample_list=sample_list,
    # )

    # fig_path = f"{plots_dir}/adjacency.pdf"
    # write_fig(fig_path, adjacency_fig, transparent=True, bbox_inches="tight")
    # NB end run_parse_n_load::parse_visium.

    pause()

    # NB by construction, require normal spots (based on baf to determine baseline).
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

    # NB triggers summary for initial clones, per single_X=1, etc; drop return.
    merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index_baf,
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # clone_id = get_clone_assignment(coords, initial_clone_index_baf)
    assignment = pd.Series(
        [f"clone {x}" for x in get_clone_assignment(coords, initial_clone_index_baf)]
    )

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
    write_fig(
        f"{plots_dir}/initial_clones_spatial.pdf",
        initial_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    logger.info(
        "Solving hmm & hmrf for copy states and clone assignment assuming baf only."
    )

    # NB zero transcript counts for all segments/spots, to drop rdr dependence
    #    of the likelihood.
    #
    # TODO can drop zero of single_X?  would be useful ...
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    # TODO utilize <BLOCK COUNTS> data structure instead of single_X, etc.
    # TODO baf_res
    res = run_core_inference(
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
        smooth_mat=smooth_mat,  # TODO HACK FINAL
        adjacency_mat=adjacency_mat,
        sample_ids=sample_ids,
        sample_list=sample_list,
        max_iter_outer=config.hmrf.max_iter_outer,
        hmmclass=hmm_nophasing,  # NB {hmm_nophasing, hmm_phased, hmm_nophasing_jax}
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
        propagate_hmm_param_errors=False,
        deconcatenate_clones=False,
    )
    # TODO
    # res.lock()

    # TODO FINAL HACK?
    # res, _ = reindex_clones(res, posterior=None, single_tumor_prop=None)

    logger.info(f"Given single_X.shape={single_X.shape}, solved for res=\n{res}")
    logger.info(
        f"Inferred {len(np.unique(res['new_assignment']))} clones given baf data."
    )

    # NB new pseduo-bulk given new assignment of spots to clones.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(res["new_assignment"], np.unique(res["new_assignment"])),
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # TODO HACK DEPRECATE? replicates tumor_prop for N clones.
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
    write_fig(
        f"{output_dir}/plots/bafonly_clones_spatial.pdf",
        bafonly_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )
    """
    bafonly_clones_genomic = plot_clones_genomic_raw(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(res["new_assignment"], np.unique(res["new_assignment"])),
        lengths,
        res=res,
        single_tumor_prop=None,
        sample_list=sample_list,
    )
    """

    bafonly_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=get_clone_indices(
            res["new_assignment"], np.unique(res["new_assignment"])
        ),
        res_combine=res,
        single_tumor_prop=None,
        sample_list=sample_list,
        palette_name="chisel_single",  # NB integer state lookup, no (A,B).
    )

    # NB inferred per-clone copy number profiles from baf-only run.
    write_fig(
        f"{plots_dir}/bafonly_clones_genomic.pdf",
        bafonly_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # NB a shallow copy.
    merged_res = res.copy()

    # NB merge similar clones based on Neyman-Pearson statistic.

    """
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
    """

    logger.info(
        f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data after neyman-pearson merge."
    )

    # NB merge according to min. number of spots per clone criterion;  single_X has dynamic shape (n_segments, 2, n_spots).
    n_obs = single_X.shape[0]
    min_umicount_thresholds = (
        n_obs * config.hmrf.min_avgumi_per_clone
    )  # MAGIC 31_420 SNP UMIs

    _, merged_res = merge_by_minspots(
        merged_res["new_assignment"],
        merged_res,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=min_umicount_thresholds,
        single_tumor_prop=single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(
        f"Inferred {len(np.unique(merged_res['new_assignment']))} clones given baf data after min spots merge."
    )

    # TODO
    # merged_res.lock()

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
    write_fig(
        f"{output_dir}/plots/merged_bafonly_clones_spatial.pdf",
        merged_bafonly_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    """
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
    )
    """

    merged_bafonly_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=get_clone_indices(
            merged_res["new_assignment"], np.unique(merged_res["new_assignment"])
        ),
        res_combine=merged_res,
        single_tumor_prop=None,
        sample_list=sample_list,
    )

    # NB inferred copy number profiles from baf-only run, after neyman-pearson "model selection"
    write_fig(
        f"{plots_dir}/merged_bafonly_clones_genomic.pdf",
        merged_bafonly_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # TODO construct for df_clone_label
    #
    # NB construct data frame with assigned clone label for all samples.
    df_clone_label = construct_df_clone_label(
        barcodes, coords, merged_res["new_assignment"], single_tumor_prop
    )

    write_tsv(
        f"{output_dir}/baf_clone_labels.tsv",
        df_clone_label,
        header=True,
        index=True,
        index_label="barcode",
        prefix="baf inferred clone labels",
    )

    # NB single_X has dynamic shape (n_segments, 2, n_spots), according to spot and
    #    segment filtering / assumed segmentation.
    #
    # TODO?  preserve segmentation, but mask emission via baf read depth or normal baseline?
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

    # TODO currently used for normal spot determination.
    # NB contains __model baf profiles__, accounted for baf-derived phase switching.
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

    #
    # =================================================================================
    # clone assignment and copy number profile refinement with gene transcripts / umis
    # =================================================================================
    #

    logger.runtime_phase = "normal candidate determination"

    # NB normal candidates (per-spot boolean) with baf only.
    normal_candidate = determine_normal_candidates(
        config,
        merged_res,
        merged_baf_profiles,
        single_X,
        copy_single_X_rdr,
        smooth_mat,
        single_tumor_prop=None,
    )

    pause()

    # TODO HACK returns umi information for refinment run with umis.
    single_X[:, 0, :] = copy_single_X_rdr

    # NB filter out genomic segments with potential allele-specific
    #    expression based on normal spot candidates;
    #
    # TODO normal mis-classification lead to dropped segments due to
    #      identifying CNAs as allele-specific expression.
    normal_idx = np.where(normal_candidate)[0]

    (
        df_gene_snp,
        (
            lengths,
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
        ),
    ) = normal_baf_bin_filter(
        df_gene_snp,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        config.phasing.nu,
        config.phasing.logphase_shift,
        normal_idx,
        config.references.geneticmap_file,
    )

    log_sitewise_transmat = get_sitewise_transmat(
        segment_key="bin_id",
        df_gene_snp=df_gene_snp,
        geneticmap_file=config.references.geneticmap_file,
        nu=config.phasing.nu,
        logphase_shift=config.phasing.logphase_shift,
    )

    # NB table of per-bin intervals with set(genes) and set(sites).
    df_bin_info = binned_gene_snp(df_gene_snp)

    # NB update to post-normal filtering single_X.
    copy_single_X_rdr = single_X[:, 0, :]

    # TODO likely removes high RDR (-only) states in simulations?
    #
    # NB zeros out high-umi differentially expressed genes,
    #    which may bias rdr estimates.
    if config.quality.filter_normal_diffexp:
        copy_single_X_rdr = filter_normal_diffexp(
            exp_counts,
            df_bin_info,
            normal_candidate,
            sample_list=sample_list,
            sample_ids=sample_ids,
        )
    else:
        logger.warning(f"Assuming no filter for normal differential expression.")

    pause()

    # TODO HACK >>>>>>  do not filter, but merge segments with insufficient normal umi counts.
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

    df_bin_info = binned_gene_snp(df_gene_snp)

    # TODO separate transmat.
    phase_indicator = np.ones(single_X.shape[0])

    # NB new segmentation and associated counts given normal candidate-based
    #    __filtering__ of baf-derived segments.
    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
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

    log_sitewise_transmat = get_sitewise_transmat(
        segment_key="bin_id",
        df_gene_snp=df_gene_snp,
        geneticmap_file=config.references.geneticmap_file,
        nu=config.phasing.nu,
        logphase_shift=config.phasing.logphase_shift,
    )

    copy_single_X_rdr = single_X[:, 0, :]
    # <<<<<<<<<<<<

    # NB >>>>>  zeros single_X_rdr entries with insufficient normal baseline counts,
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

    logger.runtime_phase = "baf/rdr clone & copy state inference"

    logger.info(
        f"Refinining {n_baf_clones} baf-identified clones with umi data assuming n_clones_rdr={config.hmrf.n_clones_rdr}"
    )

    # TODO HACK  >>>>>>>>
    if config.annotation.clone_label is not None:
        global_initial_clone_index = known_clone_assignment
    else:
        initial_rdr_clone_assignment, onehot_allowed_clones, total_clones = (
            initialize_rdr_clone_refininement(
                merged_baf_assignment=merged_baf_assignment,
                coords=coords,
                single_total_bb_RD=single_total_bb_RD,
                n_obs=single_X.shape[0],
                config=config,
            )
        )

        global_initial_clone_index = [
            np.where(initial_rdr_clone_assignment == c)[0] for c in range(total_clones)
        ]

    res_combine = run_core_inference(
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop if single_tumor_prop is not None else None,
        global_initial_clone_index,
        n_states=config.hmm.n_states,
        prefix=None,
        coords=coords,
        log_sitewise_transmat=log_sitewise_transmat,
        smooth_mat=smooth_mat,  # TODO HACK FINAL
        adjacency_mat=adjacency_mat,
        sample_ids=sample_ids,
        sample_list=sample_list,
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
        # onehot_allowed_clones=None,
        propagate_hmm_param_errors=False,
        deconcatenate_clones=True,
    )

    logger.info(f"Solved for res_combine=\n{res_combine}")

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        get_clone_indices(
            res_combine["new_assignment"], np.unique(res_combine["new_assignment"])
        ),
        single_tumor_prop if single_tumor_prop is not None else None,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # TODO HACK
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    rdr_baf_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run.
    write_fig(
        f"{output_dir}/plots/rdr_baf_clones_spatial.pdf",
        rdr_baf_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    rdr_baf_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=get_clone_indices(
            res_combine["new_assignment"], np.unique(res_combine["new_assignment"])
        ),
        res_combine=res_combine,
        single_tumor_prop=None,
        sample_list=sample_list,
        palette_name="chisel_single",  # NB integer state lookup, no (A,B).
    )

    # NB inferred per-clone copy number profiles from baf-only run.
    write_fig(
        f"{plots_dir}/rdr_baf_clones_genomic.pdf",
        rdr_baf_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    pause()

    # NB a shallow copy.
    merged_res_combine = res_combine.copy()

    """
    # NB merge similar clones based on Neyman-Pearson statistic.
    if config.hmrf.np_merge:
        _, merged_res_combine = neyman_pearson_similarity(
            X,
            base_nb_mean,
            total_bb_RD,
            res_combine,
            threshold=config.hmm.np_threshold,
            minlength=config.hmm.np_eventminlen,
            params="sp",
            tumor_prop=tumor_prop,
            hmmclass=hmm_nophasing,
        )
    """

    logger.info(
        f"Inferred {len(np.unique(merged_res_combine['new_assignment']))} clones given rdr-baf data after neyman-pearson merge."
    )

    # NB merge according to min. number of spots per clone criterion;  single_X has dynamic shape (n_segments, 2, n_spots).
    n_obs = single_X.shape[0]
    min_umicount_thresholds = (
        n_obs * config.hmrf.min_avgumi_per_clone
    )  # MAGIC 31_420 SNP UMIs

    _, merged_res_combine = merge_by_minspots(
        merged_res_combine["new_assignment"],
        merged_res_combine,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=min_umicount_thresholds,
        single_tumor_prop=single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(
        f"Inferred {len(np.unique(merged_res_combine['new_assignment']))} clones given rdr-baf data after min spots merge."
    )

    # TODO HACK
    assignment = pd.Series([f"clone {x}" for x in merged_res_combine["new_assignment"]])
    merged_rdr_baf_clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
    )

    # NB inferred clones from baf-only run, after neyman-pearson "model selection"
    write_fig(
        f"{output_dir}/plots/merged_rdr_baf_clones_spatial.pdf",
        merged_rdr_baf_clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    merged_rdr_baf_clones_genomic = plot_clones_genomic(
        df_cnv=None,
        lengths=lengths,
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        clone_index=get_clone_indices(
            merged_res_combine["new_assignment"],
            np.unique(merged_res_combine["new_assignment"]),
        ),
        res_combine=merged_res_combine,
        single_tumor_prop=None,
        sample_list=sample_list,
    )

    # NB inferred copy number profiles from baf-only run, after neyman-pearson "model selection"
    write_fig(
        f"{plots_dir}/merged_rdr_baf_clones_genomic.pdf",
        merged_rdr_baf_clones_genomic,
        transparent=True,
        bbox_inches="tight",
    )

    # TODO prev_assignment renaming.
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

    # NB re-order clones such that the index of the most-normal clone is 0.
    res_combine, _ = reindex_clones(res_combine, posterior=None, single_tumor_prop=None)
    res_combine.lock()

    final_clone_ids, final_clone_counts = np.unique(
        res_combine["new_assignment"], return_counts=True
    )

    # TODO stronger 0 .. N?
    assert 0 in final_clone_ids, "Normal clone (0) absent from final clone ids."

    logger.info(
        f"Inferred final clones=\n{final_clone_ids}\nwith fractions=\n{final_clone_counts/np.sum(final_clone_counts)}."
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

    #
    # =========================================================================================
    # integer copy number determination gived inferrered per-state (rdr,baf) and assumed ploidy.
    # =========================================================================================
    #

    logger.runtime_phase = "integer copy number determination"

    # >>>>>
    # >>>>>  TODO updated res_combine keys for integer copies, and clone assignment according to unique inferred states.
    # >>>>>

    # NB assumed ploidy for integer copy number problem, expects e.g. "diploid", "triploid", "tetraploid"
    medfix = [""] + [pp for pp in config.int_copy_num.ploidy.split(",")]

    int_ploidy_map = {"diploid": 2, "triploid": 3, "tetraploid": 4}

    # NB assumed ploidy for integer copy number problem; result is e.g. [None, 2, 3, 4] for ploidy="diploid,triploid,tetraploid".
    int_ploidy = [None] + [
        int_ploidy_map[key] for key in config.int_copy_num.ploidy.split(",")
    ]

    # TODO solution for each ploidy, enumerated by "o".
    for o, max_medploidy in enumerate(int_ploidy):
        logger.info(
            f"Solving integer copy number problem for max_medploidy={max_medploidy}."
        )

        # NB A/B integer copy number per genome segment, per state and per gene, refreshed for each max. ploidy.
        allele_specific_copy, state_cnv, df_genelevel_cnv = [], [], None

        # NB pseudobulk for each of the final clones.
        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            [
                np.where(res_combine["new_assignment"] == cid)[0]  # TODO
                for cid in final_clone_ids
            ],
            single_tumor_prop,
            threshold=config.hmrf.tumorprop_threshold,
        )

        # NB loop over clone (given max. ploidy).
        for s, cid in enumerate(final_clone_ids):
            if np.sum(base_nb_mean[:, s]) == 0:
                logger.warning("Final clone {cid} has no assigned transcripts.")
                continue

            this_pred_cnv = res_combine["pred_cnv"][:, s]

            # TODO HACK log state usage.
            us, cnts = np.unique(this_pred_cnv, return_counts=True)

            logger.info(
                f"Found state usage for clone {cid}:\n{pd.DataFrame({'state': us, 'counts': cnts})}"
            )

            # NB adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1.
            lambd = base_nb_mean[:, s] / np.sum(base_nb_mean[:, s])

            # NB scales inferred log_mu for this clone according to the library expression.
            idx = s if res_combine["new_log_mu"].shape[1] > 1 else 0

            # TODO no correction for impacy of cnas on library size?
            # adjusted_log_mu = (
            #     np.log(
            #         np.exp(res_combine["new_log_mu"][:, idx])
            #         / np.sum(
            #             lambd * np.exp(res_combine["new_log_mu"][this_pred_cnv, idx])
            #         )
            #     )
            #     if config.run.legacy
            #     else res_combine["new_log_mu"][:, idx]
            # )  # TODO HACK BUG?

            # TODO HACK BUG?
            adjusted_log_mu = res_combine["new_log_mu"][:, idx]

            logger.info(
                f"For clone {cid}, normalized log mu to sum_bin lambda * np.exp(log_mu) = 1.; yielding new mu=\n{np.exp(adjusted_log_mu)}\ngiven mu=\n{np.exp(res_combine["new_log_mu"][:, idx])}."
            )

            # TODO finalize integer copy number determination.
            #
            # NB converts inferred (rdr, baf) profiles for this clone to integer copy numbers given (max) ploidy assumption
            #    and fixed normal state as (1,1).
            if max_medploidy is not None:
                best_integer_copies, loss, best_ploidy = (
                    hill_climbing_integer_copynumber_oneclone(
                        adjusted_log_mu,
                        base_nb_mean[:, s],
                        res_combine["new_p_binom"][:, idx],
                        this_pred_cnv,
                        max_medploidy=max_medploidy,
                    )
                )
            else:
                (
                    best_integer_copies,
                    loss,
                    best_ploidy,
                ) = hill_climbing_integer_copynumber_fixdiploid_milp(
                    adjusted_log_mu,
                    base_nb_mean[:, s],
                    res_combine["new_p_binom"][:, idx],
                    this_pred_cnv,
                    nonbalance_bafdist=config.int_copy_num.nonbalance_bafdist,
                    nondiploid_rdrdist=config.int_copy_num.nondiploid_rdrdist,
                    # min_prop_threshold=0.02,  # MAGIC
                )

                # TODO HACK
                # finding_distate_failed = True
                # continue

            logger.info(
                f"Solved for (max. med ploidy, clone) = ({max_medploidy}, {s}) with integer copy number loss = {loss:.4e} and best ploidy = {best_ploidy}"
            )

            for name, data in zip(
                ("Z", "logmu", "p", "A", "B"),
                [
                    this_pred_cnv,  # NB best _REAL_ (not integer) copy states for each clone and each ploidy.
                    res_combine["new_log_mu"][
                        this_pred_cnv, idx
                    ],  # NB best model read depth for each clone and each ploidy.
                    res_combine["new_p_binom"][
                        this_pred_cnv, idx
                    ],  # NB best model baf for each clone and each ploidy.
                    best_integer_copies[
                        this_pred_cnv, 0
                    ],  # NB best integer A-copies for each clone and each ploidy.
                    best_integer_copies[
                        this_pred_cnv, 1
                    ],  # NB best integer B-copies for each clone and each ploidy.
                ],
            ):
                allele_specific_copy.append(
                    pd.DataFrame(
                        data.reshape(1, -1),
                        index=[f"clone{cid} {name}"],
                        columns=np.arange(n_obs),
                    )
                )

            for name, data in zip(
                ("logmu", "p", "A", "B"),
                [
                    res_combine["new_log_mu"][
                        :, idx
                    ],  # NB best per-state read depth for each clone and ploidy.
                    res_combine["new_p_binom"][
                        :, idx
                    ],  # NB best per-state baf for each clone and ploidy.
                    best_integer_copies[
                        :, 0
                    ],  # NB best per-state integer A-copies for each clone and ploidy.
                    best_integer_copies[
                        :, 1
                    ],  # NB best per-state integer B-copies for each clone and ploidy.
                ],
            ):
                state_cnv.append(
                    pd.DataFrame(
                        data.reshape(-1, 1),
                        columns=[f"clone{cid} {name}"],
                        index=np.arange(config.hmm.n_states),
                    )
                )

            df_genes = df_gene_snp[df_gene_snp.is_interval]
            bin_ids = df_genes["bin_id"].to_numpy(dtype=int)

            clone_copies = best_integer_copies[res_combine["pred_cnv"][:, s]]

            tmpdf = pd.DataFrame(
                {
                    "gene": df_genes.gene,
                    f"clone{s} A": clone_copies[bin_ids, 0],
                    f"clone{s} B": clone_copies[bin_ids, 1],
                }
            ).set_index("gene")

            # NB join the temporary dataframe with the existing gene-level copy number dataframe,
            #    i.e. if a subsequent clone or ploidy.
            if df_genelevel_cnv is None:
                df_genelevel_cnv = copy.copy(
                    tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int)
                )
            else:
                df_genelevel_cnv = df_genelevel_cnv.join(
                    tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int)
                )

            pause()

        # NB <<<<<< end of loop over clones.

        # NB complete loop over clones, assumed a ploidy constraint.
        if len(state_cnv) == 0:
            logger.warning(f"Found empty state integer copy numbers for clone{s}!")
            continue

        # NB write the gene-level integer copies for this assumed ploidy constraint.
        write_tsv(
            f"{output_dir}/cnv{medfix[o]}_genelevel.tsv",
            df_genelevel_cnv,
            header=True,
            index=True,
        )

        # NB output genome segment-level copy number with
        #    best integer copies for each clone and each ploidy.
        df_seglevel_cnv = df_bin_info[["CHR", "START", "END"]].join(
            pd.concat(allele_specific_copy).T
        )

        # TODO
        mask = df_seglevel_cnv.filter(regex=r" [AB]$").ne(1).any(axis=1)

        # NB display the segment-level copy number for the selected segments.
        with pd.option_context(
            "display.expand_frame_repr",
            False,
            "display.max_columns",
            None,
            "display.width",
            100_000,
            "display.max_colwidth",
            None,
        ):
            logger.debug(
                "Solved for integer copy numbers @ segments:\n%s",
                df_seglevel_cnv[mask].to_string(index=False),
            )

        # NB write integer copies for the current genome segmentation and this ploidy constraint.
        write_tsv(
            f"{output_dir}/cnv{medfix[o]}_seglevel.tsv",
            df_seglevel_cnv,
            header=True,
            index=False,
        )

        # NB output per-state integer copy numbers.
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

        # NB write integer copies for the inferred states and this ploidy constraint.
        write_tsv(
            f"{output_dir}/cnv{medfix[o]}_perstate.tsv",
            state_cnv,
            header=True,
            index=False,
        )

        # copy_states_fig = plot_copy_states(state_cnv)
        # write_fig(
        #     f"{plots_dir}/copy_states{medfix[o]}.pdf",
        #     copy_states_fig,
        #     transparent=True,
        #     bbox_inches="tight",
        # )

        # TODO HACK first ploidy constraint only;
        break

    logger.runtime_phase = "finalize"

    # NB complete inner loop over clones, and parent loop of assumed ploidy.
    #    i.e. currently assuming the last of the possible ploidy constraints,
    #         for instance "tetraploid"
    #
    # TODO could be before integer copy number solution; no dependency on integer copy number results.
    #
    # TODO constructor given barcodes, coords, new_assignment, single_tumor_prop.
    df_clone_label = construct_df_clone_label(
        barcodes, coords, res_combine["new_assignment"], single_tumor_prop
    )

    # NB does not depend on assumed ploidy.
    write_tsv(
        f"{output_dir}/clone_labels.tsv",
        df_clone_label,
        header=True,
        index=True,
        index_label="barcode",
        prefix="inferred clone labels",
    )

    # NB assumes a ploidy constraint, currently defaults to last, e.g. "tetraploid".
    real_rdr_baf_fig = plot_clones_genomic(
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        res_combine=res_combine,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        chrtext_shift=-0.3,
    )

    # TODO assumes a ploidy constraint.
    write_fig(
        f"{output_dir}/plots/real_clones_genomic.pdf",
        real_rdr_baf_fig,
        transparent=True,
        bbox_inches="tight",
    )

    # NB assumes a ploidy constraint, currently defaults to last, e.g. "tetraploid".
    rdr_baf_fig = plot_clones_genomic(
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        df_cnv=df_seglevel_cnv,  # segment level: chr, start, end, real states (Z), A/B copies, & model (log_mu, p_binom) for each clone.
        res_combine=res_combine,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        chrtext_shift=-0.3,
    )

    # TODO assumes a ploidy constraint.
    write_fig(
        f"{output_dir}/plots/clones_genomic.pdf",
        rdr_baf_fig,
        transparent=True,
        bbox_inches="tight",
    )

    # TODO issue when indexing of initial clones incompatible/bigger than final clones.
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

    # TODO
    # write_fig(f"{output_dir}/plots/initial_clones_genomic.pdf", initial_rdr_baf_fig, transparent=True, bbox_inches="tight")

    # TODO UGH enumerates, rather than actual label.
    clone_index = [
        np.where(res_combine["new_assignment"] == c)[0]
        for c, _ in enumerate(final_clone_ids)
    ]

    # clone_index = get_clone_indices(res_combine["new_assignment"], final_clone_ids)

    # NB create pseudobulk for each clone.
    X, base_nb_mean, total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        clone_index,
        single_tumor_prop,
    )

    # NB clones fig., assumes no ploidy constraint.
    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    write_fig(
        f"{output_dir}/plots/clones_spatial.pdf",
        clones_fig,
        transparent=True,
        bbox_inches="tight",
    )

    fig_copy_number_profile = plot_copy_number_profile(
        df_seglevel_cnv,  # segment level: chr, start, end, real states (Z), A/B copies, & model (log_mu, p_binom) for each clone.
    )

    write_fig(
        f"{plots_dir}/copy_number_profile.pdf",
        fig_copy_number_profile,
        transparent=True,
        bbox_inches="tight",
    )

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
        help="Override configuration keys with dot notation, e.g. -o paths.sample_sheet=/path/to/sheet.csv",
    )

    args = parser.parse_args()

    run_cnaster(args.config_path, over_rides=args.over_rides)


if __name__ == "__main__":
    main()
