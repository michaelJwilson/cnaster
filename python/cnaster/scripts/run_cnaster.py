import argparse
import copy
import time
import logging

from networkx import config
import numpy as np
import pandas as pd
import scipy
import functools
from pathlib import Path
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
from cnaster.icm import icm_sweep, unpack_adjacency
from cnaster.hmrf_utils import cast_csr
from cnaster.io import load_input_data
from cnaster.omics import (
    assign_initial_blocks,
    create_bin_ranges,
    form_gene_snp_table,
    get_sitewise_transmat,
    summarize_blocks,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnaster.phasing import initial_phase_given_partition
from cnaster.spatial import (
    fixed_rectangle_partition,
    initialize_clones,
    multislice_adjacency,
    rectangle_initialize_initial_clone,
    sufficient_umis_initial_clone,
    anisotropic_exponential_decay_adjacency,
    choose_adjacency_by_readcounts,
    renormalize_adjacency_mat,
)
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
from cnaster.sim import load_tables_to_matrices
from cnaster.hmm import pipeline_baum_welch
from cnaster.hmm_initialize import plot_cna_mixture
from cnaster.utils import merge_dicts, write_tsv, write_fig
from cnaster.integer_copy import (
    hill_climbing_integer_copynumber_oneclone,
    hill_climbing_integer_copynumber_fixdiploid,
)
from cnaster.plotting import plot_clones_genomic, plot_clones_spatial
from collections import defaultdict

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


def run_cnaster(config_path, over_rides=None):
    logger.info("----  Welcome to cnaster  ----")

    config = YAMLConfig.from_file(config_path)
    config.over_ride(over_rides)
    config.issue_warnings()
    
    logger.info(f"Read configuration:\n{config}")

    set_global_config(config)
    """
    (
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
    ) = load_tables_to_matrices()

    original_single_X = single_X.copy()
    
    # TODO HACK check against above.
    smooth_mat, adjacency_mat = choose_adjacency_by_readcounts(
        coords, single_total_bb_RD
    )
    smooth_mat.eliminate_zeros()
    adjacency_mat.eliminate_zeros()

    logger.info(f"Found adjacency matrix:\n{adjacency_mat}")

    # NB renormalize cumulative edge weight to median in each case.
    adjacency_mat = renormalize_adjacency_mat(adjacency_mat)
    """

    # NB start run_parse_n_load::parse_visium::load_joint_data
    #    adata: (barcode x gene) transcripts ('count') + 'tumor_annotation' + 'X_pos' + slice ('sample').
    #    cell_snp_Aallele: haplotype H0 counts (barcode x snp).
    #    cell_snp_Ballele: haplotype H1 counts (barcode x snp).
    #    unique_snp_ids: {contig}_{pos}_{ref}_{alt} for all snps.
    #    across_slice_adjacency_mat: ...
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
        min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
    )

    # NB e.g. 'AAACAAGTATCTCCCA-1_HT112C1-U1' currently.
    barcodes = adata.obs.index
    sample_list = [adata.obs["sample"].iloc[0]]

    # NB loop through rows (barcodes x samples) and collect sample names;
    #    assumes sorted by sample and is unique in this case.
    for i in range(1, adata.shape[0]):
        if adata.obs["sample"].iloc[i] != sample_list[-1]:
            sample_list.append(adata.obs["sample"].iloc[i])

    # NB e.g. HT112C1-U1.
    logger.info(f"Found {len(sample_list)} unique samples, e.g. {sample_list[:3]}")

    # NB array: assigns to each transcript row (barcode x sample) unique index according to sample names.
    sample_ids = -np.ones(adata.shape[0], dtype=int)

    for s, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        sample_ids[index] = s

    assert np.all(
        sample_ids >= 0
    ), f"Failed to assign unique integer to all samples in list. Bug?"

    if config.preprocessing.tumorprop_file is not None:
        logger.info(
            f"Reading pre-processed tumorprop file={config.preprocessing.tumorprop_file}"
        )

        df_tumorprop = pd.read_csv(
            config.preprocessing.tumorprop_file, sep="\t", header=0, index_col=0
        )

        df_tumorprop = df_tumorprop[["Tumor"]]
        df_tumorprop.columns = ["tumor_proportion"]

        assert np.all(
            adata.obs.index == df_tumorprop.index
        ), "Detected mis-alignment of AnnData & tumor prop. barcode/sample ordering."

        adata.obs = adata.obs.join(df_tumorprop)

        single_tumor_prop = adata.obs["tumor_proportion"]
    else:
        logger.info(f"No (pre-processed) tumorprop. file provided.")
        single_tumor_prop = None

    # NB parse_visium::combine_gene_snps
    #    chr, start, end, snp_id, gene, is_interval (is_gene).
    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )

    # NB parse_visium::create_haplotype_block_ranges
    df_gene_snp = assign_initial_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        initial_min_umi=15, # MAGIC
    )

    # NB num. of blocks per contig; SN-based H0 and H0+H1 counts block; total UMIs per block.
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

    # NB 1D array of expected phase error rate.
    log_sitewise_transmat = get_sitewise_transmat(
        df_gene_snp,
        config.references.geneticmap_file,
        config.phasing.nu,
        config.phasing.logphase_shift,
    )

    # NB (x,y) per spot.
    coords = adata.obsm["X_pos"]

    # NB known annotation.
    if config.annotation.clone_label is not None:
        logger.warning("Assuming known clone labels")

        clone_id = (
            pd.read_csv(config.annotation.clone_label, sep="\t", index_col=0)["labels"]
            .str.replace("clone_", "")
            .str.replace("normal", "-1")
            .astype(int)
            .to_numpy()
        )
        clone_id += 1

        initial_clone_index_baf = [
            np.where(clone_id == xx)[0] for xx in np.unique(clone_id)
        ]

        # TODO HACK!
        initial_clone_for_phasing = initial_clone_index_baf

        known_rdr_normal = np.sum(single_X[:, 0, (clone_id == 0)], axis=1)  

        bidx_inconfident = np.where(known_rdr_normal < config.quality.min_normal_count_perbin)[0] 
        known_rdr_normal[bidx_inconfident] = 0

        # NB normalized.
        known_rdr_normal = known_rdr_normal / np.sum(known_rdr_normal)

        spots_coverage = np.sum(single_X[:, 0, :], axis=0)

        known_single_base_nb_mean = known_rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)
    else:
        initial_clone_index_baf = None
        known_single_base_nb_mean = None

        # NB equivalent to parse_visium::perform_partition
        # TODO (requires paste).
        initial_clone_for_phasing = initialize_clones(
            coords,
            sample_ids,  # NB for all spots in all slices.
            x_part=config.phasing.npart_phasing,
            y_part=config.phasing.npart_phasing,
        )
    """
    # TODO HACK
    prephase_X, prephase_base_nb_mean, prephase_total_bb_RD, _ = merge_pseudobulk_by_index_mix(
        single_X,
        known_single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index_baf,
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # TODO HACK
    plot_cna_mixture(
        None,
        None,
        None,
        None,
        prephase_X,
        prephase_base_nb_mean,
        prephase_total_bb_RD,
        prefix="pre_phasing",
        max_rdr=10,
    )
    """

    logger.warning("Assuming (magic) five BAF states for phasing.")

    assert single_X.ndim == 3

    if config.phasing.run:
        # NB single_base_nb_mean initialized to zero - requires normal spot. determination.
        phase_indicator, refined_lengths = initial_phase_given_partition(
            single_X,
            lengths,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            initial_clone_for_phasing,
            5,  # MAGIC n_states
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

    # NB phase is None for genes and otherwise True/False for the phase of each block.
    df_gene_snp["phase"] = np.where(
        df_gene_snp.snp_id.isnull(),
        None,
        df_gene_snp.block_id.map({i: x for i, x in enumerate(phase_indicator)}),
    )

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

    logger.info(f"Recalculating counts given new phase-based bins.")

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

    # NB sparse transcript counts (spot, gene).
    exp_counts = pd.DataFrame.sparse.from_spmatrix(
        scipy.sparse.csc_matrix(adata.layers["count"]),
        index=adata.obs.index,
        columns=adata.var.index,
    )

    # NB smooth pooling matrix & distance based (exponential decay) adjacency.
    adjacency_mat, smooth_mat = multislice_adjacency(
        sample_ids,
        sample_list,
        coords,
        single_total_bb_RD,  # NEGLECTED?
        exp_counts,  # NEGLECTED?
        across_slice_adjacency_mat,
        construct_adjacency_method=config.hmrf.construct_adjacency_method,
        maxspots_pooling=config.hmrf.maxspots_pooling,
        construct_adjacency_w=config.hmrf.construct_adjacency_w,
        unit_xsquared=config.hmrf.unit_xsquared,
        unit_ysquared=config.hmrf.unit_ysquared,
    )

    # TODO table_bininfo? table_rdrbaf? table_meta?
    # NB end run_parse_n_load::parse_visium.

    # NB by construction, require normal spots (based on BAF to determine baseline).
    assert np.all(single_base_nb_mean == 0)

    # TODO
    copy_single_X_rdr = copy.copy(single_X[:, 0, :])

    # NB zeros
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)

    """
    # NB non-contiguous assignment of clones to an unequal grid partitioning
    #    of input coordinates.
    initial_clone_index_baf, clone_id = rectangle_initialize_initial_clone(
        coords, config.hmrf.n_clones, random_state=0
    )
    """
    """
    # TODO HACK
    x_part = y_part = 4
    initial_clone_index_baf, clone_id = fixed_rectangle_partition(
        coords, x_part, y_part, single_tumor_prop=None, threshold=0.5
    )
    """
    
    # TODO HACK? adata.layers["count"]
    if initial_clone_index_baf is None:
        initial_clone_index_baf, clone_id, spot_umi_counts = sufficient_umis_initial_clone(
            coords,
            single_X[:,0,:],
            sample_list,
            sample_ids,
            500_000,
            random_state=int(config.hmrf.random_state),
        )
    """
    adj_list = cast_csr(adjacency_mat)
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

    single_llf = posterior = np.zeros((single_X.shape[-1], config.hmrf.n_clones))
    
    # NB high is exclusive.
    clone_id = np.random.randint(0, high=config.hmrf.n_clones, size=single_X.shape[-1])

    icm_sweep(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        clone_id,
        config.hmrf.spatial_weight,
        posterior,
        tol=0.01,
        log_persample_weights=None,
        sample_ids=None,
        cost_zeropoint=0.0,
        temp=1.0,
    )
    
    initial_clone_index_baf = [np.where(clone_id == i)[0] for i in range(config.hmrf.n_clones)]
    """
    # NB trigger summary for initial clones, per single_X=1 etc.
    merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index_baf,
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # NB construct clone labels.
    df_clone_label = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]}, index=barcodes
    )

    # NB barcodes is the index.
    df_clone_label.insert(0, "sample_id", df_clone_label.index.str.split("_").str[-1])

    # TODO assert aligned?
    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    df_clone_label["clone_label"] = clone_id

    # TODO HACK
    # df_clone_label["UMIs"] = spot_umi_counts

    # NB cannot sort before barcode-ordered assignments etc!
    df_clone_label = df_clone_label.groupby("sample_id", group_keys=False).apply(
        lambda g: g.sort_values(["x", "y"])
    )

    output_dir = f"{config.paths.output_dir}/clone{config.hmrf.n_clones}.{config.hmrf.n_clones_rdr}_rectangle{config.hmrf.random_state}_w{config.hmrf.spatial_weight:.1f}/"

    if not (poutput_dir := Path(output_dir)).exists():
        poutput_dir.mkdir(exist_ok=True)

    plots_dir = f"{output_dir}/plots/"

    if not (pplots_dir := Path(plots_dir)).exists():
        pplots_dir.mkdir(exist_ok=True)

    opath = f"{output_dir}/initial_clone_labels.tsv"
    logger.info(f"Writing initial clone labels to {opath},\n{df_clone_label.head()}")
    
    write_tsv(opath, df_clone_label, header=True, index=True, index_label="barcode")

    # TODO HACK
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

    fig_path = f"{output_dir}/plots/initial_clones_spatial.pdf"
    write_fig(fig_path, initial_clones_fig, transparent=True, bbox_inches="tight")
    
    logger.info(
        "Solving HMM & HMRF for copy states and clone assignment with BAF only."
    )

    # NB baf-only run: zero transcript counts for all segments/spots.
    # TODO can drop zero of single_X?  would be useful ...
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    res = hmrfmix_concatenate_pipeline(
        None,
        None,
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        initial_clone_index_baf,
        config.hmm.n_states,
        log_sitewise_transmat,
        smooth_mat=smooth_mat,
        adjacency_mat=adjacency_mat,
        sample_ids=sample_ids,
        max_iter_outer=config.hmrf.max_iter_outer,
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
        max_iter=config.hmm.max_iter,
        tol=config.hmm.tol,
        spatial_weight=config.hmrf.spatial_weight,
        tumorprop_threshold=config.hmrf.tumorprop_threshold,
    )

    # NB number of bins/segments/blocks
    n_obs = single_X.shape[0]

    # NB new pseduo-bulk given new assignment of spots to clones.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        [
            np.where(res["new_assignment"] == c)[0]
            for c in np.sort(np.unique(res["new_assignment"]))
        ],
        single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # TODO
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

    fig_path = f"{output_dir}/plots/bafonly_clones_spatial.pdf"
    write_fig(fig_path, bafonly_clones_fig, transparent=True, bbox_inches="tight")

    if config.hmrf.np_merge:
        # NB merge similar clones based on Neyman-Pearson
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

    _, merged_res = merge_by_minspots(
        merged_res["new_assignment"],
        merged_res,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=n_obs * config.hmrf.min_avgumi_per_clone,
        single_tumor_prop=single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
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

    fig_path = f"{output_dir}/plots/merged_bafonly_clones_spatial.pdf"
    write_fig(
        fig_path, merged_bafonly_clones_fig, transparent=True, bbox_inches="tight"
    )

    # NB construct clone labels.
    df_clone_label = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]}, index=barcodes
    )

    # NB barcodes is the index.
    df_clone_label.insert(0, "sample_id", df_clone_label.index.str.split("_").str[-1])

    # TODO assert aligned?
    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    df_clone_label["clone_label"] = merged_res["new_assignment"]

    # NB cannot sort before barcode-ordered assignments etc!
    df_clone_label = df_clone_label.groupby("sample_id", group_keys=False).apply(
        lambda g: g.sort_values(["x", "y"])
    )

    opath = f"{output_dir}/baf_clone_labels.tsv"

    logger.info(
        f"Writing baf inferred clone labels to {opath},\n{df_clone_label.head()}"
    )

    write_tsv(opath, df_clone_label, header=True, index=True, index_label="barcode")

    # TODO
    n_obs = single_X.shape[0]

    # NB clone assignment based on BAF only, after merging similar clones.
    merged_baf_assignment = copy.copy(merged_res["new_assignment"])
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # NB MAP copy state.
    pred = np.argmax(merged_res["log_gamma"], axis=0)

    # NB split into per-contig list, vs single concatenated array.
    pred = np.array(
        [pred[(c * n_obs) : (c * n_obs + n_obs)] for c in range(n_baf_clones)]
    )

    logger.info(
        f"Found {100. * np.mean(pred[:, :] < config.hmm.n_states)}% of BAF-only copy states to have phase 0."
    )

    # DEPRECATE?  baf-only clones are determined with hmm_nophasing.
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

    logger.info(f"Determining normal spots based on BAF-only clones.")

    # NB no input files for barcodes of normal spots, or tumor proportion per spot.
    if (config.preprocessing.normalidx_file is None) and (
        config.preprocessing.tumorprop_file is None
    ):
        EPS_BAF = 0.05  # MAGIC
        PERCENT_NORMAL = 40  # MAGIC

        logger.info(
            f"Identifying normal spots based on estimated BAF given EPS_BAF={EPS_BAF} and PERCENT_NORMAL={PERCENT_NORMAL}."
        )

        # NB sum deviations > EPS_BAF from 0.5 along the genome for each clone; pick normal as minimum deviation.
        baf_deviations = np.sum(
            np.maximum(np.abs(merged_baf_profiles - 0.5) - EPS_BAF, 0), axis=1
        )
        id_nearnormal_clone = np.argmin(baf_deviations)

        logger.info(
            f"Found clone {id_nearnormal_clone} to be the most normal-like given BAF deviations."
        )

        # NB measure the standard deviation of log-transformed, smoothed transcript counts for each spot.
        vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)

        prior_stdthreshold = np.inf

        while True:
            # NB spots assigned to the normal-like clone AND 40% with smallest BAF deviation from 0.5;
            stdthreshold = np.percentile(
                vec_stds[merged_res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (
                merged_res["new_assignment"] == id_nearnormal_clone
            )
            
            if config.run.legacy and ( np.sum(copy_single_X_rdr[:, (normal_candidate == True)]) > 200 * single_X.shape[0]):
                logger.info(
                    f"Assumed legacy normal spot allocation for {PERCENT_NORMAL}[%] normal spots"
                )
                break
            
            elif stdthreshold > 1.5 * prior_stdthreshold:  # MAGIC
                logger.info(
                    f"Determined {PERCENT_NORMAL}% normal spots with sufficient UMIs, assigned to normal like clone."
                )
                logger.info(
                    f"BAF-clone breakdown:\n{np.unique(merged_res["new_assignment"][normal_candidate], return_counts=True)}"
                )
                break
            elif PERCENT_NORMAL == 100:
                logger.warning(
                    f"All {np.count_nonzero(merged_res["new_assignment"] == id_nearnormal_clone)} spots for clone {id_nearnormal_clone} considered to be normal."
                )
                break

            PERCENT_NORMAL += 10

    elif config.preprocessing.normalidx_file is not None:
        # single_base_nb_mean has already been added in loading data step.
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
    else:
        assert single_tumor_prop is not None

        logger.info(f"Identifying normal spots based on provided tumor proportion.")

        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            # NB suggests 0 is perfectly normal and otherwise measures tumor proportion, sensibly!
            normal_candidate = single_tumor_prop < prop_threshold

            if (
                np.sum(copy_single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Determined normal spots with sufficient UMIs based on input tumor proportion @ prop_threshold={prop_threshold}"
                )
                break
        else:
            logger.warning(
                f"Failed to determine normal spots with sufficient UMIs based on input tumor proportion."
            )

    index_normal = np.where(normal_candidate)[0]

    # TODO HACK
    single_X[:, 0, :] = copy_single_X_rdr

    # NB filter out genomic segments with potential allele-specific expression based on normal spot candidates.
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

    # NB new bin info.
    df_bininfo = binned_gene_snp(df_gene_snp)

    copy_single_X_rdr = single_X[:, 0, :]
    
    # NB filter out high-UMI DE genes, which may bias RDR estimates.
    if config.quality.filter_normal_diffexp:
        copy_single_X_rdr, _ = filter_normal_diffexp(
            exp_counts,
            df_bininfo,
            normal_candidate,
            sample_list=sample_list,
            sample_ids=sample_ids,
        )

    # TODO HACK
    elif False:
        logger.warning(f"Assuming basic normal differential expression.")

        normal_gene_counts = np.sum(adata.layers["count"][normal_candidate, :], axis=0)
        tumor_gene_counts = np.sum(adata.layers["count"][~normal_candidate, :], axis=0)

        scaled_normal_gene_counts = normal_gene_counts * len(normal_candidate) / np.count_nonzero(normal_candidate)

        # TODO HACK both ways?
        diff_exp_thres = 6. # MAGIC
        exp_diff_exp = (tumor_gene_counts / scaled_normal_gene_counts > diff_exp_thres) # | (scaled_normal_gene_counts / tumor_gene_counts > diff_exp_thres)

        total_original_umis = adata.layers["count"].sum()
        
        # TODO assumes single_X etc will be re-calculated downstream.
        adata.layers["count"][:, exp_diff_exp] = 0.0
        
        total_original_umis_retained = adata.layers["count"].sum()
                   
        logger.info(f"Zeroed {100. * np.mean(exp_diff_exp):.3f} [%] of genes with {(1. - total_original_umis_retained/total_original_umis):.3f} of UMIs estimated to be driven by differential expression.")
                   
    # TODO CHECK?
    else:
        logger.warning(f"Assuming no filter for normal differential expression.")
        
    """    
    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="bin_id",
        normal_candidates=normal_candidate,
    )
    """

    # TODO HACK >>>>>>  do not filter, but merge segments, with insufficient normal umi counts.
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
        key="bin_id"
    )

    # NB new bin info.                                                                                                                                                                                              
    df_bininfo = binned_gene_snp(df_gene_snp)

    # TODO separate transmat.
    phase_indicator = np.ones(single_X.shape[0])
    
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
    
    # NB >>>>>  determine normal baseline expression.
    logger.info(
        f"Found sparsity of normal spot set={100. * np.mean(copy_single_X_rdr[:, (normal_candidate == True)]) == 0.:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    rdr_normal = np.sum(copy_single_X_rdr[:, (normal_candidate == True)], axis=1)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100. * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )

    pct_list = [1, 5, 25, 50, 75] + list(range(90, 101, 1))
    rdr_pcts = np.percentile(rdr_normal, pct_list)
    per_segment_std = copy_single_X_rdr[:, (normal_candidate == True)].std(axis=1)
    std_pcts = np.percentile(per_segment_std, pct_list)

    logger.info(f"For percentiles={pct_list}, rdr_normal percentiles=\n{rdr_pcts}\nand per-segment std across normal candidates percentiles=\n{std_pcts}")

    # TODO HACK
    high_std_idx = np.where(per_segment_std > std_pcts[-2])[0]
    high_rdr_idx = np.where(rdr_normal > rdr_pcts[-2])[0]
    bidx_inconfident = np.unique(
        np.concatenate([bidx_inconfident, high_std_idx, high_rdr_idx])
    )

    # NB where normal transcript count < config.quality.min_normal_count_perbin, zero.
    rdr_normal[bidx_inconfident] = 0

    # NB normalized.
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # NB avoid ill-defined distributions if normal has 0 count in that bin, assuming clone
    #    should have no expression if normal does not - true for copy number models.
    copy_single_X_rdr[bidx_inconfident, :] = 0

    # NB replicate and normalize rdr_normal to the per-spot total transcripts, T_n.
    spots_coverage = np.sum(copy_single_X_rdr, axis=0)

    copy_single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    # NB adding back RDR signal
    single_X[:, 0, :] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean
    n_obs = single_X.shape[0]
    # <<<<<

    logger.info(
        f"Refinining {n_baf_clones} BAF identified clones with RDR data assuming n_clones_rdr={config.hmrf.n_clones_rdr}"
    )

    clone_res = {}

    for bafc in range(n_baf_clones):
        logger.info(f"Solving for BAF clone {bafc}/{n_baf_clones}.")

        prefix = f"clone{bafc}"

        # NB spots assigned to this BAF-only clone (after merging based on Neyman-Pearson similarity).
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        # NB min. b-allele read count (equivalent to 20 per spot) on pseudobulk to split clones.
        # TODO split will be on RDR, seems an odd requirement?
        if np.sum(single_total_bb_RD[:, idx_spots]) < 20 * single_X.shape[0]:
            logger.warning(
                f"Skipping RDR refinment of BAF identified clone {bafc} as too few snp-covering UMIs ({np.sum(single_total_bb_RD[:, idx_spots])}/{20 * single_X.shape[0]})!"
            )
            continue

        """
        # NB initialize new set of clones within this BAF identified clone.
        # TODO tumor_prop, i.e. _mix.
        initial_clone_index, _ = rectangle_initialize_initial_clone(
            coords[idx_spots],
            config.hmrf.n_clones_rdr,
            random_state=0,  # TODO HACK.
        )
        """

        # TODO HACK?  splits each BAF clone along the x direction.
        x_part, y_part = config.hmrf.n_clones_rdr, 1
        
        initial_clone_index, _ = fixed_rectangle_partition(
            coords[idx_spots],
            x_part,
            y_part,
        )

        initial_assignment = np.zeros(len(idx_spots), dtype=int)

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

        # NB hmrf + hmm with RDR data.
        new_clone_res = hmrfmix_concatenate_pipeline(
            None,  # NB outdir
            None,  # NB prefix
            single_X[:, :, idx_spots],
            lengths,
            single_base_nb_mean[
                :, idx_spots
            ],  # NB per-spot replications normalized to T_n.
            single_total_bb_RD[:, idx_spots],
            single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
            initial_clone_index,  # NB
            n_states=config.hmm.n_states,
            log_sitewise_transmat=log_sitewise_transmat,
            smooth_mat=smooth_mat[np.ix_(idx_spots, idx_spots)],
            adjacency_mat=adjacency_mat[np.ix_(idx_spots, idx_spots)],
            sample_ids=copy_slice_sample_ids,
            max_iter_outer=config.hmrf.max_iter_outer,
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
            max_iter=config.hmm.max_iter,
            tol=config.hmm.tol,
            spatial_weight=config.hmrf.spatial_weight,
            tumorprop_threshold=config.hmrf.tumorprop_threshold,
            init_p_binom=None, # merged_res["new_p_binom"],  # HACK? BAF states known from BAF-only run. # TODO init mus conditioned on known baf.
            init_log_mu=None, # np.zeros_like(merged_res["new_p_binom"]),  # HACK? BAF states known from BAF-only run.  # TODO init mus conditioned on known baf.
        )

        clone_res[prefix] = merge_dicts(clone_res[prefix], new_clone_res)

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
            logger.info(f"Clone {bafc} was not split by RDR.")

            # NB clone id.
            c, n_merged_clones = res["new_assignment"][0], 1

            # NB merging is a null op.
            merged_res = copy.copy(res)
            merged_res["new_assignment"] = np.zeros(len(idx_spots), dtype=int)

            # NB first case for hmm_sitewise with phased states, otherwise n_states.
            try:
                log_gamma = res["log_gamma"][
                    :, (c * n_obs) : (c * n_obs + n_obs)
                ].reshape((2 * config.hmm.n_states, n_obs, 1))
            except:
                log_gamma = res["log_gamma"][
                    :, (c * n_obs) : (c * n_obs + n_obs)
                ].reshape((config.hmm.n_states, n_obs, 1))

            # NB MAP copy state.
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
                logger.warning("No Neyman-Pearson merging applied to RDR identified clones.")
                merged_res = res.copy()

            merging_groups, merged_res = merge_by_minspots(
                merged_res["new_assignment"],
                merged_res,
                single_total_bb_RD[:, idx_spots],
                min_spots_thresholds=config.hmrf.min_spots_per_clone,
                min_umicount_thresholds=n_obs * config.hmrf.min_avgumi_per_clone,
                single_tumor_prop=(
                    single_tumor_prop[idx_spots]
                    if single_tumor_prop is not None
                    else None
                ),
                threshold=config.hmrf.tumorprop_threshold,
            )

            # NB num. of rdr-split clones within baf clone after merging.
            n_merged_clones = len(merging_groups)
            tmp = copy.copy(merged_res["new_assignment"])

            # NB compute posterior using the newly merged pseudobulk
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
                single_X[:, :, idx_spots],
                single_base_nb_mean[:, idx_spots],
                single_total_bb_RD[:, idx_spots],
                [
                    np.where(merged_res["new_assignment"] == c)[0]
                    for c in range(n_merged_clones)
                ],
                single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
                threshold=config.hmrf.tumorprop_threshold,
            )

            # NB recompute copy states and clone profiles based on new pseudobulk.
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

            merged_res["new_assignment"] = copy.copy(tmp)

            # NB combined only between similar states in the RDR split clones.
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

        # TODO safe merge.
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

        offset_clone += n_merged_clones

    logger.info(
        f"Assuming max. alpha dispersion between clones given current:\n{res_combine['new_alphas']}"
    )
    logger.info(
        f"Assuming min. tau dispersion between clones given current:\n{res_combine['new_taus']}"
    )

    # HACK broadcast max. dispersion across all clones.
    res_combine["new_alphas"][:, :] = np.max(res_combine["new_alphas"])

    # HACK broadcast min. dispersion across all clones; tau is total pseduocounts for BAF
    #      min. is least significant.
    res_combine["new_taus"][:, :] = np.min(res_combine["new_taus"])

    n_final_clones = len(np.unique(res_combine["prev_assignment"]))

    logger.info(f"Inferred {n_final_clones} clones given BAF+RDR data.")

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

    # NB final re-assignment across all spots using estimated copy number states.
    if config.preprocessing.tumorprop_file is None:
        if config.hmrf.nodepotential == "max":
            pred = np.vstack(
                [
                    np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
                    for c in range(res_combine["log_gamma"].shape[2])
                ]
            ).T

            # TODO!!
            new_assignment, _, total_llf, posterior = aggr_hmrf_reassignment(
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
            raise RuntimeError()
    else:
        if config.hmrf.nodepotential == "max":
            pred = np.vstack(
                [
                    np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
                    for c in range(res_combine["log_gamma"].shape[2])
                ]
            ).T

            (
                new_assignment,
                single_llf,
                total_llf,
                posterior,
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
        else:
            raise RuntimeError()

    # NB total Potts likelihood given final copy states and clone assignment.
    res_combine["total_llf"] = total_llf
    res_combine["new_assignment"] = new_assignment

    # NB re-order clones such that the normal clone is always 0.
    res_combine, posterior = reindex_clones(res_combine, posterior, single_tumor_prop)

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

    # NB infer integer allele-specific copy numbers
    final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))

    # NB add normal clone as 0 if not present
    if 0 not in final_clone_ids:
        final_clone_ids = np.append(0, final_clone_ids)

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
                logger.warning("Final clone {cid} has no assigned transcripts!")
                continue

            # lambd = base_nb_mean[:, s] / np.sum(base_nb_mean[:, s])
            this_pred_cnv = res_combine["pred_cnv"][:, s]

            # NB log state usage
            us, cnts = np.unique(this_pred_cnv, return_counts=True)

            # TODO HACK
            logger.info(
                f"Found state usage for clone {cid}:\n{pd.DataFrame({'state': us, 'counts': cnts})}"
            )

            # NB adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1.
            adjusted_log_mu = np.log(
                np.exp(res_combine["new_log_mu"][:, s])
                / np.sum(np.exp(res_combine["new_log_mu"][this_pred_cnv, s]) * lambd)
            ) if False else res_combine["new_log_mu"][:, s] # TODO HACK BUG?
            
            logger.info(f"For clone {cid}, normalized log mu to sum_bin lambda * np.exp(log_mu) = 1.; yielding new mu=\n{np.exp(adjusted_log_mu)}\ngiven mu=\n{np.exp(res_combine["new_log_mu"][:, s])}.")

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
                    best_integer_copies[this_pred_cnv, 0].reshape(
                        1, -1
                    ),
                    index=[f"clone{cid} A"],
                    columns=np.arange(n_obs),
                )
            )
            allele_specific_copy.append(
                pd.DataFrame(
                    best_integer_copies[this_pred_cnv, 1].reshape(
                        1, -1
                    ),
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
        mask = (
            df_seglevel_cnv[a_cols].ne(1) |
            df_seglevel_cnv[b_cols].ne(1)
        ).any(axis=1)

        with pd.option_context(
            "display.expand_frame_repr", False,
            "display.max_columns", None,
            "display.width", 100000,
            "display.max_colwidth", None,
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
            "display.expand_frame_repr", False,
            "display.max_columns", None,
            "display.width", None,
            "display.max_colwidth", None,
        ):
            logger.info(
                "Solved for integer copy numbers @ states:\n%s",
                state_cnv.to_string(index=False),
            )

        opath = f"{output_dir}/cnv{medfix[o]}_perstate.tsv"
        write_tsv(opath, state_cnv, header=True, index=False)

    # NB construct clone labels.
    df_clone_label = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]}, index=barcodes
    )

    # NB barcodes is the index.
    df_clone_label.insert(0, "sample_id", df_clone_label.index.str.split("_").str[-1])

    # TODO assert aligned?
    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    df_clone_label["clone_label"] = res_combine["new_assignment"]

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
        # chrtext_shift=-0.3,
        base_height=3.2,
        # pointsize=15,
        # linewidth=1,
        palette_name="chisel",
    )

    # TODO
    fig_path = f"{output_dir}/plots/clones_genomic.pdf"
    write_fig(fig_path, rdr_baf_fig, transparent=True, bbox_inches="tight")

    # TODO issue when indexing of initial clones incompatiable/bigger than final clones.
    initial_rdr_baf_fig = plot_clones_genomic(
        df_seglevel_cnv,
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        res_combine,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        clone_ids=None,
        clone_index=initial_clone_index_baf,
        remove_xticks=True,
        base_height=3.2,
        palette_name="chisel",
    )

    # TODO
    fig_path = f"{output_dir}/plots/initial_clones_genomic.pdf"
    write_fig(fig_path, initial_rdr_baf_fig, transparent=True, bbox_inches="tight")

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
