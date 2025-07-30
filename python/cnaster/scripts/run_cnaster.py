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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("cnaster.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_cnaster(config_path):
    start_time = time.time()

    config = YAMLConfig.from_file(config_path)

    set_global_config(config)

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

    logger.info(f"Forming gene & snp meta data.")
    
    df_gene_snp = form_gene_snp_table(
        unique_snp_ids, config.references.hgtable_file, adata
    )
    
    logger.info(f"Assigning initial blocks")
    
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
        1.0e-3,  # MAGIC tol
        threshold=config.hmrf.tumorprop_threshold,
    )

    logger.info(f"Solved for initial phase given Eagle & BAF in {(time.time() - start_time)/60.:.2f} minutes.")
    
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

    # TODO
    copy_single_X_rdr = copy.copy(single_X[:, 0, :])
    copy_single_base_nb_mean = copy.copy(single_base_nb_mean)

    # NB baf-only run;
    single_X[:, 0, :] = 0
    single_base_nb_mean[:, :] = 0

    logger.info("Solving for multislice_adjaceny.")
    
    initial_clone_index = rectangle_initialize_initial_clone(
        coords, config.hmrf.n_clones, random_state=0
    )

    logger.info("Solving HMM+HMRF for copy state and clones.")
    
    res = hmrfmix_concatenate_pipeline(
        None,
        None,
        single_X,
        lengths,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        initial_clone_index,
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

    # TODO HACK
    n_obs = single_X.shape[0]

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

    logger.info("Merging similar clones assuming Neyman-Pearson")

    merging_groups, merged_res = neyman_pearson_similarity(
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

    merging_groups, merged_res = merge_by_minspots(
        merged_res["new_assignment"],
        merged_res,
        single_total_bb_RD,
        min_spots_thresholds=config.hmrf.min_spots_per_clone,
        min_umicount_thresholds=n_obs * config.hmrf.min_avgumi_per_clone,
        single_tumor_prop=single_tumor_prop,
        threshold=config.hmrf.tumorprop_threshold,
    )

    # TODO
    # NB re-phase
    n_obs = single_X.shape[0]

    merged_baf_assignment = copy.copy(merged_res["new_assignment"])
    n_baf_clones = len(np.unique(merged_baf_assignment))
    pred = np.argmax(merged_res["log_gamma"], axis=0)
    pred = np.array(
        [pred[(c * n_obs) : (c * n_obs + n_obs)] for c in range(n_baf_clones)]
    )
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

    logger.info(f"Refinining {n_baf_clones} BAF identified clones with RDR data assuming n_clones_rdr={config.hmrf.n_clones_rdr}")
    
    # NB refine BAF-identified clones
    if (config.preprocessing.normalidx_file is None) and (
        config.preprocessing.tumorprop_file is None
    ):
        EPS_BAF = 0.05  # MAGIC
        PERCENT_NORMAL = 40  # MAGIC

        logger.info(f"Identifying normal spots based on estimated BAF.")

        vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
        id_nearnormal_clone = np.argmin(
            np.sum(np.maximum(np.abs(merged_baf_profiles - 0.5) - EPS_BAF, 0), axis=1)
        )

        while True:
            stdthreshold = np.percentile(
                vec_stds[merged_res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (
                merged_res["new_assignment"] == id_nearnormal_clone
            )
            if (
                np.sum(copy_single_X_rdr[:, (normal_candidate == True)])
                > single_X.shape[0] * 200
                or PERCENT_NORMAL == 100
            ):
                break
            PERCENT_NORMAL += 10

    elif config.preprocessing.normalidx_file is not None:
        # single_base_nb_mean has already been added in loading data step.
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
    else:
        logger.info(f"Identifying normal spots based on provided tumor proportion.")

        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            normal_candidate = single_tumor_prop < prop_threshold

            # TODO
            if (
                np.sum(copy_single_X_rdr[:, (normal_candidate == True)])
                > single_X.shape[0] * 200
            ):
                break

    index_normal = np.where(normal_candidate)[0]

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

    # TODO HACK
    df_bininfo = binned_gene_snp(df_gene_snp)

    copy_single_X_rdr = copy.copy(single_X[:, 0, :])

    # NB filter out high-UMI DE genes, which may bias RDR estimates
    copy_single_X_rdr, _ = filter_normal_diffexp(
        exp_counts,
        df_bininfo,
        normal_candidate,
        sample_list=sample_list,
        sample_ids=sample_ids,
    )

    MIN_NORMAL_COUNT_PERBIN = 20  # MAGIC
    bidx_inconfident = np.where(
        np.sum(copy_single_X_rdr[:, (normal_candidate == True)], axis=1)
        < MIN_NORMAL_COUNT_PERBIN
    )[0]
    rdr_normal = np.sum(copy_single_X_rdr[:, (normal_candidate == True)], axis=1)
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # NB avoid ill-defined distributions if normal has 0 count in that bin.
    copy_single_X_rdr[bidx_inconfident, :] = 0
    copy_single_base_nb_mean = rdr_normal.reshape(-1, 1) @ np.sum(
        copy_single_X_rdr, axis=0
    ).reshape(1, -1)

    # NB adding back RDR signal
    single_X[:, 0, :] = copy_single_X_rdr
    single_base_nb_mean = copy_single_base_nb_mean
    n_obs = single_X.shape[0]

    clone_res = {}

    for bafc in range(n_baf_clones):
        logger.info(f"Solving for BAF clone {bafc}/{n_baf_clones}.")

        prefix = f"clone{bafc}"
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        # NB min. b-allele read count on pseudobulk to split clones
        if np.sum(single_total_bb_RD[:, idx_spots]) < 20 * single_X.shape[0]:
            logger.warning(f"TODO")
            continue

        # TODO tumor_prop, i.e. _mix.
        initial_clone_index = rectangle_initialize_initial_clone(
            coords[idx_spots],
            config.hmrf.n_clones_rdr,
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
        )

    # TODO HACK
    logger.info(f"Combining results across clones.")

    res_combine = {"prev_assignment": np.zeros(single_X.shape[2], dtype=int)}
    offset_clone = 0

    # NB Neyman-Pearson and min. spot merging across RDR redefined clones.
    for bafc in range(n_baf_clones):
        prefix = f"clone{bafc}"
        res = clone_res[prefix]

        # TODO HACK?
        idx_spots = np.where(barcodes.isin(res["barcodes"]))[0]

        # NB
        if len(np.unique(res["new_assignment"])) == 1:
            # TODO HACK
            logger.warning(f"Found a single clone.")

            n_merged_clones = 1
            c = res["new_assignment"][0]
            merged_res = copy.copy(res)
            merged_res["new_assignment"] = np.zeros(len(idx_spots), dtype=int)

            try:
                log_gamma = res["log_gamma"][
                    :, (c * n_obs) : (c * n_obs + n_obs)
                ].reshape((2 * config["n_states"], n_obs, 1))
            except:
                log_gamma = res["log_gamma"][
                    :, (c * n_obs) : (c * n_obs + n_obs)
                ].reshape((config["n_states"], n_obs, 1))

            pred_cnv = res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)].reshape(
                (-1, 1)
            )
        else:
            X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
                single_X[:, :, idx_spots],
                single_base_nb_mean[:, idx_spots],
                single_total_bb_RD[:, idx_spots],
                [
                    np.where(res["new_assignment"] == c)[0]
                    for c in np.sort(np.unique(res["new_assignment"]))
                ],
                single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
                threshold=config.hmrf.tumorprop_threshold,
            )

            if tumor_prop is not None:
                tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)

            merging_groups, merged_res = neyman_pearson_similarity(
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

            # TODO check merge_by_minspots logging.
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

            # NB compute posterior using the newly merged pseudobulk
            n_merged_clones = len(merging_groups)
            tmp = copy.copy(merged_res["new_assignment"])

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
                merge_threshold=0.1,
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
                        "new_taus": np.hstack(
                            n_merged_clones * [merged_res["new_taus"]]
                        ),
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

    # HACK assume dispersions are the same across all clones (max?)
    res_combine["new_alphas"][:, :] = np.max(res_combine["new_alphas"])

    # HACK BUG!? assume dispersions are the same across all clones (min??)
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

    # NB final re-assignment across all clones using estimated copy number states.
    if config.preprocessing.tumorprop_file is None:
        if config.hmrf.nodepotential == "max":
            pred = np.vstack(
                [
                    np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
                    for c in range(res_combine["log_gamma"].shape[2])
                ]
            ).T
            new_assignment, single_llf, total_llf, posterior = aggr_hmrf_reassignment(
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
        elif config.hmrf.nodepotential == "weighted_sum":
            (
                new_assignment,
                single_llf,
                total_llf,
                posterior,
            ) = hmrf_reassignment_posterior(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                res_combine,
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

        elif config.hmrf.nodepotential == "weighted_sum":
            (
                new_assignment,
                single_llf,
                total_llf,
                posterior,
            ) = hmrfmix_reassignment_posterior(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                single_tumor_prop,
                res_combine,
                smooth_mat,
                adjacency_mat,
                res_combine["prev_assignment"],
                copy.copy(sample_ids),
                log_persample_weights,
                spatial_weight=config.hmrf.spatial_weight,
                hmmclass=hmm_nophasing,
                return_posterior=True,
            )

    res_combine["total_llf"] = total_llf
    res_combine["new_assignment"] = new_assignment

    # NB re-order clones such that normal clones are always clone 0.
    res_combine, posterior = reindex_clones(res_combine, posterior, single_tumor_prop)

    # TODO new_log_startprob - add to res_combine above.
    for key in ["new_log_mu", "new_alphas", "new_p_binom", "new_taus", "pred_cnv"]:
        logger.info(f"Solved for {key}:\n{res_combine[key]}")

    # NB ----  infer integer allele-specific copy number  ----
    final_clone_ids = np.sort(np.unique(res_combine["new_assignment"]))

    nonempty_clone_ids = copy.copy(final_clone_ids)

    # NB add normal clone as 0 if not present
    if 0 not in final_clone_ids:
        final_clone_ids = np.append(0, final_clone_ids)

    # NB ploidy
    medfix = ["", "_diploid", "_triploid", "_tetraploid"]

    for o, max_medploidy in enumerate([None, 2, 3, 4]):
        logger.info(
            f"Solving integer copy number problem for max_medploidy={max_medploidy}"
        )

        # NB A/B integer copy number per bin and per state
        allele_specific_copy, state_cnv = [], []
        df_genelevel_cnv = None

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
                logger.warning("TODO")
                continue

            # NB adjust log_mu such that sum_bin lambda * np.exp(log_mu) = 1.
            lambd = base_nb_mean[:, s] / np.sum(base_nb_mean[:, s])
            this_pred_cnv = res_combine["pred_cnv"][:, s]
            adjusted_log_mu = np.log(
                np.exp(res_combine["new_log_mu"][:, s])
                / np.sum(np.exp(res_combine["new_log_mu"][this_pred_cnv, s]) * lambd)
            )
            if max_medploidy is not None:
                best_integer_copies, loss = hill_climbing_integer_copynumber_oneclone(
                    adjusted_log_mu,
                    base_nb_mean[:, s],
                    res_combine["new_p_binom"][:, s],
                    this_pred_cnv,
                    max_medploidy=max_medploidy,
                )
            else:
                try:
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
                    )
                except:
                    try:
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
                            min_prop_threshold=0.02,  # MAGIC
                        )
                    except:
                        finding_distate_failed = True
                        continue

            logger.info(
                f"Solved for (max. med ploidy, clone) = ({max_medploidy}, {s}) with integer copy number loss = {loss:.4e}"
            )

            allele_specific_copy.append(
                pd.DataFrame(
                    best_integer_copies[res_combine["pred_cnv"][:, s], 0].reshape(
                        1, -1
                    ),
                    index=[f"clone{cid} A"],
                    columns=np.arange(n_obs),
                )
            )
            allele_specific_copy.append(
                pd.DataFrame(
                    best_integer_copies[res_combine["pred_cnv"][:, s], 1].reshape(
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
            continue

        # NB output gene-level copy number
        # df_genelevel_cnv.to_csv(
        #    f"{outdir}/cnv{medfix[o]}_genelevel.tsv", header=True, index=True, sep="\t"
        # )

        logger.info(
            f"Solved for integer copy numbers @ genes:\n{df_genelevel_cnv.head()}"
        )

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

        # df_seglevel_cnv.to_csv(
        #    f"{outdir}/cnv{medfix[o]}_seglevel.tsv", header=True, index=False, sep="\t"
        # )

        logger.info(
            f"Solved for integer copy numbers @ segments:\n{df_seglevel_cnv.head()}"
        )

        # NB output per-state copy number
        state_cnv = functools.reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="inner"
            ),
            state_cnv,
        )

        logger.info(f"Solved for integer copy numbers @ states:\n{state_cnv}")

        # state_cnv.to_csv(
        # f"{outdir}/cnv{medfix[o]}_perstate.tsv", header=True, index=False, sep="\t"
        # )

    # TODO CHECK
    df_clone_label = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]}, index=barcodes
    )

    if config.preprocessing.tumorprop_file is not None:
        df_clone_label["tumor_proportion"] = single_tumor_prop

    df_clone_label["clone_label"] = res_combine["new_assignment"]

    opath = f"{config.paths.output_dir}/clone_labels.tsv"

    logger.info(f"Writing inferred clone labels to {opath},\n{df_clone_label.head()}")

    # TODO HACK
    # df_clone_label.to_csv(f"{outdir}/clone_labels.tsv", header=True, index=True, sep="\t")

    rdr_baf_fig = plot_clones_genomic(
        df_seglevel_cnv,
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        res_combine,
        single_tumor_prop=single_tumor_prop,
        clone_ids=None,
        remove_xticks=True,
        rdr_ylim=5,
        chrtext_shift=-0.3,
        base_height=3.2,
        pointsize=15,
        linewidth=1,
        palette="chisel",
    )

    # TODO
    fig_path = f"{config.paths.output_dir}/plots/clones_genomic.pdf"
    logger.info(f"Writing clones genomic fig. to {fig_path}")
    # rdr_baf_fig.savefig(fig_path, transparent=True, bbox_inches="tight")

    assignment = pd.Series([f"clone {x}" for x in res_combine["new_assignment"]])
    clones_fig = plot_clones_spatial(
        coords,
        assignment,
        single_tumor_prop=single_tumor_prop,
        sample_list=sample_list,
        sample_ids=sample_ids,
        base_width=4,
        base_height=3,
        palette="Set2",
    )

    # TODO
    fig_path = f"{config.paths.output_dir}/plots/clones_spatial.pdf"
    logger.info(f"Writing clones spatial fig. to {fig_path}")
    # clones_fig.savefig(fig_path, transparent=True, bbox_inches="tight")

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
