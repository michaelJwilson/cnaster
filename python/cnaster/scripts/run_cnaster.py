import argparse
import copy
import logging

import numpy as np
import pandas as pd
import scipy
from cnaster.config import YAMLConfig, set_global_config
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf import hmrfmix_concatenate_pipeline, merge_by_minspots
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
from cnaster.neyman_pearson import neyman_pearson_similarity
from cnaster.normal_spot import (
    normal_baf_bin_filter,
    filter_normal_diffexp,
    binned_gene_snp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def run_cnaster(config_path):
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

    # NB 1D array / list?
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
        coords, config.hmrf.n_clones, random_state=0
    )

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

    logger.info(f"Finding refinement of {n_baf_clones} BAF-identified clones.")

    for bafc in range(n_baf_clones):
        logger.info(f"Solving for BAF clone {bafc}/{n_baf_clones}.")

        prefix = f"clone{bafc}"
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        # NB min. b-allele read count on pseudobulk to split clones
        if np.sum(single_total_bb_RD[:, idx_spots]) < 20 * single_X.shape[0]:
            continue

        # TODO tumor_prop
        initial_clone_index = rectangle_initialize_initial_clone(
            coords[idx_spots],
            config.hmrf.n_clones_rdr,
            random_state=0,  # TODO HACK.
        )

        # HMRF + HMM using RDR data.
        copy_slice_sample_ids = copy.copy(sample_ids[idx_spots])

        hmrfmix_concatenate_pipeline(
            None,
            None,
            single_X[:, :, idx_spots],
            lengths,
            single_base_nb_mean[:, idx_spots],
            single_total_bb_RD[:, idx_spots],
            single_tumor_prop[idx_spots] if single_tumor_prop is not None else None,
            initial_clone_index,
            n_states=config.hmm.n_states,
            log_sitewise_transmat=log_sitewise_transmat,
            smooth_mat=smooth_mat[np.ix_(idx_spots, idx_spots)],
            adjacency_mat=adjacency_mat[np.ix_(idx_spots, idx_spots)],
            sample_ids=copy_slice_sample_ids,
            max_iter_outer=config.hmrf.max_iter_outer,  # TODO MAGIC 10
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

        # NB combine results across clones
        res_combine = {"prev_assignment": np.zeros(single_X.shape[2], dtype=int)}
        offset_clone = 0
        for bafc in range(n_baf_clones):
            prefix = f"clone{bafc}"
            allres = dict(
                np.load(
                    f"{outdir}/{prefix}_nstates{config['n_states']}_smp.npz",
                    allow_pickle=True,
                )
            )
            r = allres["num_iterations"] - 1
            res = {
                "new_log_mu": allres[f"round{r}_new_log_mu"],
                "new_alphas": allres[f"round{r}_new_alphas"],
                "new_p_binom": allres[f"round{r}_new_p_binom"],
                "new_taus": allres[f"round{r}_new_taus"],
                "new_log_startprob": allres[f"round{r}_new_log_startprob"],
                "new_log_transmat": allres[f"round{r}_new_log_transmat"],
                "log_gamma": allres[f"round{r}_log_gamma"],
                "pred_cnv": allres[f"round{r}_pred_cnv"],
                "llf": allres[f"round{r}_llf"],
                "total_llf": allres[f"round{r}_total_llf"],
                "prev_assignment": allres[f"round{r-1}_assignment"],
                "new_assignment": allres[f"round{r}_assignment"],
            }
            idx_spots = np.where(barcodes.isin(allres["barcodes"]))[0]
            if len(np.unique(res["new_assignment"])) == 1:
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
                if config["tumorprop_file"] is None:
                    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(
                        single_X[:, :, idx_spots],
                        single_base_nb_mean[:, idx_spots],
                        single_total_bb_RD[:, idx_spots],
                        [
                            np.where(res["new_assignment"] == c)[0]
                            for c in np.sort(np.unique(res["new_assignment"]))
                        ],
                    )
                    tumor_prop = None
                else:
                    X, base_nb_mean, total_bb_RD, tumor_prop = (
                        merge_pseudobulk_by_index_mix(
                            single_X[:, :, idx_spots],
                            single_base_nb_mean[:, idx_spots],
                            single_total_bb_RD[:, idx_spots],
                            [
                                np.where(res["new_assignment"] == c)[0]
                                for c in np.sort(np.unique(res["new_assignment"]))
                            ],
                            single_tumor_prop[idx_spots],
                            threshold=config["tumorprop_threshold"],
                        )
                    )
                    tumor_prop = np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
                merging_groups, merged_res = similarity_components_rdrbaf_neymanpearson(
                    X,
                    base_nb_mean,
                    total_bb_RD,
                    res,
                    threshold=config["np_threshold"],
                    minlength=config["np_eventminlen"],
                    params="smp",
                    tumor_prop=tumor_prop,
                    hmmclass=hmm_nophasing_v2,
                )
                print(f"part {bafc} merging_groups: {merging_groups}")
                #
                if config["tumorprop_file"] is None:
                    merging_groups, merged_res = merge_by_minspots(
                        merged_res["new_assignment"],
                        merged_res,
                        single_total_bb_RD[:, idx_spots],
                        min_spots_thresholds=config["min_spots_per_clone"],
                        min_umicount_thresholds=config["min_avgumi_per_clone"] * n_obs,
                    )
                else:
                    merging_groups, merged_res = merge_by_minspots(
                        merged_res["new_assignment"],
                        merged_res,
                        single_total_bb_RD[:, idx_spots],
                        min_spots_thresholds=config["min_spots_per_clone"],
                        min_umicount_thresholds=config["min_avgumi_per_clone"] * n_obs,
                        single_tumor_prop=single_tumor_prop[idx_spots],
                        threshold=config["tumorprop_threshold"],
                    )
                print(
                    f"part {bafc} merging after requiring minimum # spots: {merging_groups}"
                )
                # compute posterior using the newly merged pseudobulk
                n_merged_clones = len(merging_groups)
                tmp = copy.copy(merged_res["new_assignment"])
                if config["tumorprop_file"] is None:
                    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(
                        single_X[:, :, idx_spots],
                        single_base_nb_mean[:, idx_spots],
                        single_total_bb_RD[:, idx_spots],
                        [
                            np.where(merged_res["new_assignment"] == c)[0]
                            for c in range(n_merged_clones)
                        ],
                    )
                    tumor_prop = None
                else:
                    X, base_nb_mean, total_bb_RD, tumor_prop = (
                        merge_pseudobulk_by_index_mix(
                            single_X[:, :, idx_spots],
                            single_base_nb_mean[:, idx_spots],
                            single_total_bb_RD[:, idx_spots],
                            [
                                np.where(merged_res["new_assignment"] == c)[0]
                                for c in range(n_merged_clones)
                            ],
                            single_tumor_prop[idx_spots],
                            threshold=config["tumorprop_threshold"],
                        )
                    )
                #
                merged_res = pipeline_baum_welch(
                    None,
                    np.vstack(
                        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
                    ).T.reshape(-1, 2, 1),
                    np.tile(lengths, X.shape[2]),
                    config["n_states"],
                    base_nb_mean.flatten("F").reshape(-1, 1),
                    total_bb_RD.flatten("F").reshape(-1, 1),
                    np.tile(log_sitewise_transmat, X.shape[2]),
                    (
                        np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
                        if not tumor_prop is None
                        else None
                    ),
                    hmmclass=hmm_nophasing_v2,
                    params="smp",
                    t=config["t"],
                    random_state=config["gmm_random_state"],
                    fix_NB_dispersion=config["fix_NB_dispersion"],
                    shared_NB_dispersion=config["shared_NB_dispersion"],
                    fix_BB_dispersion=config["fix_BB_dispersion"],
                    shared_BB_dispersion=config["shared_BB_dispersion"],
                    is_diag=True,
                    init_log_mu=res["new_log_mu"],
                    init_p_binom=res["new_p_binom"],
                    init_alphas=res["new_alphas"],
                    init_taus=res["new_taus"],
                    max_iter=config["max_iter"],
                    tol=config["tol"],
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
                    hmmclass=hmm_nophasing_v2,
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
            #
            # add to res_combine
            if len(res_combine) == 1:
                res_combine.update(
                    {
                        "new_log_mu": np.hstack(
                            [merged_res["new_log_mu"]] * n_merged_clones
                        ),
                        "new_alphas": np.hstack(
                            [merged_res["new_alphas"]] * n_merged_clones
                        ),
                        "new_p_binom": np.hstack(
                            [merged_res["new_p_binom"]] * n_merged_clones
                        ),
                        "new_taus": np.hstack(
                            [merged_res["new_taus"]] * n_merged_clones
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
                            + [merged_res["new_log_mu"]] * n_merged_clones
                        ),
                        "new_alphas": np.hstack(
                            [res_combine["new_alphas"]]
                            + [merged_res["new_alphas"]] * n_merged_clones
                        ),
                        "new_p_binom": np.hstack(
                            [res_combine["new_p_binom"]]
                            + [merged_res["new_p_binom"]] * n_merged_clones
                        ),
                        "new_taus": np.hstack(
                            [res_combine["new_taus"]]
                            + [merged_res["new_taus"]] * n_merged_clones
                        ),
                        "log_gamma": np.dstack([res_combine["log_gamma"], log_gamma]),
                        "pred_cnv": np.hstack([res_combine["pred_cnv"], pred_cnv]),
                    }
                )
            res_combine["prev_assignment"][idx_spots] = (
                merged_res["new_assignment"] + offset_clone
            )
            offset_clone += n_merged_clones
        # temp: make dispersions the same across all clones
        res_combine["new_alphas"][:, :] = np.max(res_combine["new_alphas"])
        res_combine["new_taus"][:, :] = np.min(res_combine["new_taus"])
        # end temp
        n_final_clones = len(np.unique(res_combine["prev_assignment"]))
        # per-sample weights across clones
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
        # final re-assignment across all clones using estimated RDR + BAF
        if config["tumorprop_file"] is None:
            if config["nodepotential"] == "max":
                pred = np.vstack(
                    [
                        np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
                        for c in range(res_combine["log_gamma"].shape[2])
                    ]
                ).T
                new_assignment, single_llf, total_llf, posterior = (
                    aggr_hmrf_reassignment(
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
                        spatial_weight=config["spatial_weight"],
                        hmmclass=hmm_nophasing_v2,
                        return_posterior=True,
                    )
                )
            elif config["nodepotential"] == "weighted_sum":
                new_assignment, single_llf, total_llf, posterior = (
                    hmrf_reassignment_posterior(
                        single_X,
                        single_base_nb_mean,
                        single_total_bb_RD,
                        res_combine,
                        smooth_mat,
                        adjacency_mat,
                        res_combine["prev_assignment"],
                        copy.copy(sample_ids),
                        log_persample_weights,
                        spatial_weight=config["spatial_weight"],
                        hmmclass=hmm_nophasing_v2,
                        return_posterior=True,
                    )
                )
        else:
            if config["nodepotential"] == "max":
                pred = np.vstack(
                    [
                        np.argmax(res_combine["log_gamma"][:, :, c], axis=0)
                        for c in range(res_combine["log_gamma"].shape[2])
                    ]
                ).T
                new_assignment, single_llf, total_llf, posterior = (
                    aggr_hmrfmix_reassignment(
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
                        spatial_weight=config["spatial_weight"],
                        hmmclass=hmm_nophasing_v2,
                        return_posterior=True,
                    )
                )
            elif config["nodepotential"] == "weighted_sum":
                new_assignment, single_llf, total_llf, posterior = (
                    hmrfmix_reassignment_posterior(
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
                        spatial_weight=config["spatial_weight"],
                        hmmclass=hmm_nophasing_v2,
                        return_posterior=True,
                    )
                )
        res_combine["total_llf"] = total_llf
        res_combine["new_assignment"] = new_assignment
        # re-order clones such that normal clones are always clone 0
        res_combine, posterior = reorder_results(
            res_combine, posterior, single_tumor_prop
        )
        # save results
        np.savez(
            f"{outdir}/rdrbaf_final_nstates{config['n_states']}_smp.npz", **res_combine
        )
        np.save(f"{outdir}/posterior_clone_probability.npy", posterior)

    logger.info("Done.\n\n")


# NB run_cnaster config_turing.yaml
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
