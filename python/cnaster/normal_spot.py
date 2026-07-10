import ast

import anndata
import numpy as np
import scanpy as sc
import scipy
import scipy.stats
from numba import njit
from sklearn.cluster import KMeans

from cnaster.config import get_global_config, start_time
from cnaster.hmm_emission import Weighted_BetaBinom
from cnaster.hmm_utils import get_em_solver_params
from cnaster.logger import get_logger
from cnaster.spatio_genomic_counts import SpatioGenomicCounts

logger = get_logger(__name__, start_time=start_time)


def determine_normal_candidates(
    config,
    res,
    baf_profiles,
    single_X,
    single_X_rdr,
    smooth_mat,
    single_tumor_prop=None,
):
    """
    Determine normal candidate spots based on BAF profiles, tumor proportion, or provided indices.
    Returns a boolean array normal_candidate.
    """
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
            np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1
        )
        id_nearnormal_clone = np.argmin(baf_deviations)

        logger.info(
            f"Found clone {id_nearnormal_clone} to be the most normal-like given BAF deviations."
        )

        # NB measure the standard deviation of log-transformed, smoothed transcript counts for each spot.
        if smooth_mat is not None:
            vec_stds = np.std(np.log1p(single_X_rdr @ smooth_mat), axis=0)
        else:
            vec_stds = np.std(np.log1p(single_X_rdr), axis=0)

        prior_stdthreshold = np.inf

        while True:
            stdthreshold = np.percentile(
                vec_stds[res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (
                res["new_assignment"] == id_nearnormal_clone
            )

            # DEPRECATE config.run_legacy.
            if False and (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
                > 200 * single_X.shape[0]  # MAGIC
            ):
                logger.info(
                    f"Assumed legacy normal spot allocation for {PERCENT_NORMAL}[%] normal spots"
                )
                break

            elif stdthreshold > 1.5 * prior_stdthreshold:  # MAGIC
                logger.info(
                    f"Determined {PERCENT_NORMAL}% normal spots with sufficient UMIs, assigned to normal like clone."
                )
                logger.info(
                    f"BAF-clone breakdown:\n{np.unique(res['new_assignment'][normal_candidate], return_counts=True)}"
                )
                break
            elif PERCENT_NORMAL == 100:
                logger.warning(
                    f"All {np.count_nonzero(res['new_assignment'] == id_nearnormal_clone)} spots for clone {id_nearnormal_clone} considered to be normal."
                )
                break

            PERCENT_NORMAL += 10
        return normal_candidate

    elif config.preprocessing.normalidx_file is not None:
        # single_base_nb_mean has already been added in loading data step.
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
        # You may want to load normal_candidate from file here
        return None

    else:
        assert single_tumor_prop is not None

        logger.info(f"Identifying normal spots based on provided tumor proportion.")

        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            # NB suggests 0 is perfectly normal and otherwise measures tumor proportion, sensibly!
            normal_candidate = single_tumor_prop < prop_threshold

            if (
                np.sum(single_X_rdr[:, (normal_candidate == True)])
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
        return normal_candidate


def determine_normal_baseline(single_X_rdr, normal_candidate, config=None):
    if config is None:
        config = get_global_config()

    logger.info(
        f"Found sparsity of normal spot set={100. * np.mean(single_X_rdr[:, (normal_candidate == True)]) == 0.:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    # TODO normal_candidate clause
    rdr_normal = np.sum(single_X_rdr[:, (normal_candidate == True)], axis=1)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100. * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )

    # NB where normal transcript count < config.quality.min_normal_count_perbin, zero.
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # TODO copy?
    # NB avoid ill-defined distributions if normal has 0 count in that bin.
    single_X_rdr[bidx_inconfident, :] = 0

    # NB replicate and normalize rdr_normal to the per-spot total transcripts, T_n.
    spots_coverage = np.sum(single_X_rdr, axis=0)

    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    return rdr_normal, single_X_rdr, single_base_nb_mean


@njit
def compute_local_normal_mask(
    baf_profiles,  # (n_clones, n_bins)
    spot_assignments,  # (n_spots,) mapping each spot to a clone index
    single_X_rdr,  # (n_bins, n_spots) raw RNA counts
    window_size,  # int: The "n_segments" to look left and right
    eps_baf=0.05,  # float: allowed BAF wiggle room around 0.5
    baf_margin=0.01,  # float: margin to allow multiple identical clones to pass
    start_percentile=0.4,  # float: start relaxing from 40% of variance distribution
    percentile_step=0.05,  # float: step size (+5%) for relaxing transcript variance
    var_ratio_thresh=1.1,  # float: Required factor of SS_between / SS_within to stop
    amp_min=0.8,  # float: lower bound for amplitude regression
    amp_max=1.25,  # float: upper bound for amplitude regression
):
    """
    Creates a (n_bins x n_spots) boolean mask tracking which spots act as the optimal
    normal baseline for a given genomic segment.

    Identifies a single global BAF-normal clone that is unconditionally anchored.
    Additional localized clones are checked for regression amplitude consistency
    with the anchor, and iteratively merged if they fit within the variance ratio threshold.
    """
    n_clones, n_bins = baf_profiles.shape
    n_spots = single_X_rdr.shape[1]

    local_mask = np.zeros((n_bins, n_spots), dtype=np.bool_)

    # -------------------------------------------------------------------------
    # GLOBAL PRE-COMPUTE: Find global normal clone & normalize clone expressions
    # -------------------------------------------------------------------------

    # 1A. Find single global BAF normal clone
    global_baf_devs = np.zeros(n_clones)
    for c in range(n_clones):
        d = 0.0
        for i in range(n_bins):
            diff = np.abs(baf_profiles[c, i] - 0.5) - eps_baf
            if diff > 0:
                d += diff
        global_baf_devs[c] = d

    global_anchor_clone = 0
    min_global_dev = global_baf_devs[0]
    for c in range(1, n_clones):
        if global_baf_devs[c] < min_global_dev:
            min_global_dev = global_baf_devs[c]
            global_anchor_clone = c

    # 1B. Aggregate spots into pure clone pseudo-bulks and get library sizes
    clone_X_rdr = np.zeros((n_bins, n_clones), dtype=np.float64)
    clone_spot_counts = np.zeros(n_clones, dtype=np.int32)
    clone_lib_size = np.zeros(n_clones, dtype=np.float64)

    for s in range(n_spots):
        c = spot_assignments[s]
        clone_spot_counts[c] += 1
        s_sum = 0.0
        for i in range(n_bins):
            val = single_X_rdr[i, s]
            clone_X_rdr[i, c] += val
            s_sum += val
        clone_lib_size[c] += s_sum

    # Precompute mean log1p of the clone expressions
    # and normalize clone_X_rdr raw counts by library size for fast regression
    log1p_clone_rdr = np.empty_like(clone_X_rdr)
    norm_clone_rdr = np.empty_like(clone_X_rdr)

    for c in range(n_clones):
        if clone_spot_counts[c] > 0 and clone_lib_size[c] > 0:
            scale_factor = 1.0 / clone_lib_size[c]
        else:
            scale_factor = 0.0

        for i in range(n_bins):
            if clone_spot_counts[c] > 0:
                log1p_clone_rdr[i, c] = np.log1p(
                    clone_X_rdr[i, c] / clone_spot_counts[c]
                )
            else:
                log1p_clone_rdr[i, c] = 0.0

            norm_clone_rdr[i, c] = clone_X_rdr[i, c] * scale_factor

    # -------------------------------------------------------------------------
    # MAIN LOOP: Segment-by-segment localized clustering
    # -------------------------------------------------------------------------
    total_accepted_clones = 0
    for b in range(n_bins):
        w_start = max(0, b - window_size)
        w_end = min(n_bins, b + window_size + 1)
        w_len = w_end - w_start

        # 1. Start with the guaranteed global anchor
        accepted_clones = np.zeros(n_clones, dtype=np.bool_)
        if clone_spot_counts[global_anchor_clone] > 0:
            accepted_clones[global_anchor_clone] = True

        # 2. Find LOCAL diploid clones based on BAF variance (initial candidates)
        local_baf_devs = np.zeros(n_clones)
        min_local_dev = 1e9

        for c in range(n_clones):
            if clone_spot_counts[c] == 0:
                continue
            dev = 0.0
            for i in range(w_start, w_end):
                diff = np.abs(baf_profiles[c, i] - 0.5) - eps_baf
                if diff > 0:
                    dev += diff
            dev_avg = dev / w_len
            local_baf_devs[c] = dev_avg
            if dev_avg < min_local_dev:
                min_local_dev = dev_avg

        # 3. Filter candidates: check BAF margin AND check Negative Binomial Regression Proxy Amplitude
        candidate_clones = np.zeros(n_clones, dtype=np.int32)
        n_candidates = 0

        for c in range(n_clones):
            if clone_spot_counts[c] > 0 and c != global_anchor_clone:
                if local_baf_devs[c] <= min_local_dev + baf_margin:

                    # Compute pseudo-Poisson/NB weighted regression of candidate vs Global Anchor
                    # y = Candidate, x = Anchor.
                    # In count models, variance scales with mean. A 1st-order WLS sets weights ~ 1/mean.
                    sum_wxy = 0.0
                    sum_wxx = 0.0

                    for i in range(w_start, w_end):
                        x_val = norm_clone_rdr[i, global_anchor_clone]
                        y_val = norm_clone_rdr[i, c]

                        # Weight proportional to inverse expected variance
                        weight = 1.0 / (x_val + 1e-8)

                        sum_wxy += weight * x_val * y_val
                        sum_wxx += weight * x_val * x_val

                    if sum_wxx > 1e-12:
                        beta = sum_wxy / sum_wxx
                    else:
                        beta = 0.0

                    # Only accept as a candidate if amplitude difference is within 20% copy number bounds
                    if amp_min <= beta <= amp_max:
                        candidate_clones[n_candidates] = c
                        n_candidates += 1

        # 4. Calculate transcript variance specifically for surviving CLONES
        if n_candidates > 0:
            clone_stds = np.zeros(n_candidates)
            for idx in range(n_candidates):
                c = candidate_clones[idx]
                mean_val = 0.0
                for i in range(w_start, w_end):
                    mean_val += log1p_clone_rdr[i, c]
                mean_val /= w_len

                var_val = 0.0
                for i in range(w_start, w_end):
                    diff = log1p_clone_rdr[i, c] - mean_val
                    var_val += diff * diff
                var_val /= w_len
                clone_stds[idx] = np.sqrt(var_val)

            sorted_idx = np.argsort(clone_stds)

            # 5. Iteratively relax the threshold to absorb ordered clones
            total_pool_size = n_candidates + 1
            pool_stds = np.zeros(total_pool_size)

            # Get anchor variance
            mean_anchor = 0.0
            for i in range(w_start, w_end):
                mean_anchor += log1p_clone_rdr[i, global_anchor_clone]
            mean_anchor /= w_len
            var_anchor = 0.0
            for i in range(w_start, w_end):
                diff = log1p_clone_rdr[i, global_anchor_clone] - mean_anchor
                var_anchor += diff * diff
            var_anchor /= w_len

            pool_stds[0] = np.sqrt(var_anchor)
            for i in range(n_candidates):
                pool_stds[i + 1] = clone_stds[sorted_idx[i]]

            mean_all_stds = np.mean(pool_stds)

            if total_pool_size > 3:
                best_keep = total_pool_size
                percentile = start_percentile

                while percentile <= 0.95:
                    thresh_idx = int(percentile * total_pool_size)
                    if thresh_idx < 1:
                        thresh_idx = 1
                    if thresh_idx >= total_pool_size - 1:
                        thresh_idx = total_pool_size - 2

                    n_in = thresh_idx + 1
                    sum_in = 0.0
                    for i in range(n_in):
                        sum_in += pool_stds[i]
                    mean_in = sum_in / n_in

                    n_out = total_pool_size - n_in
                    sum_out = 0.0
                    for i in range(thresh_idx + 1, total_pool_size):
                        sum_out += pool_stds[i]
                    mean_out = sum_out / n_out

                    ss_w = 0.0
                    for i in range(n_in):
                        d = pool_stds[i] - mean_in
                        ss_w += d * d
                    for i in range(thresh_idx + 1, total_pool_size):
                        d = pool_stds[i] - mean_out
                        ss_w += d * d

                    d_in = mean_in - mean_all_stds
                    d_out = mean_out - mean_all_stds
                    ss_b = (n_in * d_in * d_in) + (n_out * d_out * d_out)

                    if ss_w > 1e-12:
                        if ss_b > var_ratio_thresh * ss_w:
                            best_keep = n_in
                            break

                    percentile += percentile_step

                for i in range(best_keep - 1):
                    c = candidate_clones[sorted_idx[i]]
                    accepted_clones[c] = True
            else:
                for i in range(n_candidates):
                    c = candidate_clones[i]
                    accepted_clones[c] = True

        # 6. Lock in spots from accepted clones for this segment
        accepted_count = 0
        for s in range(n_spots):
            c_idx = spot_assignments[s]
            if accepted_clones[c_idx]:
                local_mask[b, s] = True

        for c in range(n_clones):
            if accepted_clones[c]:
                accepted_count += 1

        total_accepted_clones += accepted_count

    avg_clones_included = total_accepted_clones / n_bins
    return local_mask, avg_clones_included


@njit
def _compute_baseline_core(single_X_rdr, local_normal_mask, min_normal_count_perbin):
    """
    JIT compiled core for computing baseline stats and zeroing out low-confidence
    segments in-place without generating massive intermediate matrices.
    """
    n_bins, n_spots = single_X_rdr.shape
    rdr_normal = np.zeros(n_bins, dtype=np.float64)

    # 1. Sum up normal spots per bin utilizing the 2D sliding-window mask
    for i in range(n_bins):
        b_sum = 0.0
        for j in range(n_spots):
            if local_normal_mask[i, j]:
                b_sum += single_X_rdr[i, j]
        rdr_normal[i] = b_sum

    # 2. Filter unconfident bins directly in place
    for i in range(n_bins):
        if rdr_normal[i] < min_normal_count_perbin:
            rdr_normal[i] = 0.0
            # Avoid ill-defined distributions if normal baseline falls short
            for j in range(n_spots):
                single_X_rdr[i, j] = 0.0

    # 3. Normalize rdr_normal to create a probability distribution across genomic segments
    rdr_sum = 0.0
    for i in range(n_bins):
        rdr_sum += rdr_normal[i]

    if rdr_sum > 0:
        for i in range(n_bins):
            rdr_normal[i] /= rdr_sum

    # 4. Calculate total spot coverage AFTER zeroing out unconfident segments
    spots_coverage = np.zeros(n_spots, dtype=np.float64)
    for j in range(n_spots):
        s_sum = 0.0
        for i in range(n_bins):
            s_sum += single_X_rdr[i, j]
        spots_coverage[j] = s_sum

    return rdr_normal, spots_coverage


def determine_local_normal_baseline(
    config,
    res,
    baf_profiles,
    single_X,
    single_X_rdr,
    smooth_mat=None,  # Kept to preserve exact caller parameter index compatibility
    single_tumor_prop=None,
    window_size=100,
):
    """
    Unified entrypoint replacing determine_normal_candidates and determine_normal_baseline.
    Derives segment-specific normal cells dynamically and solves for the global expected
    baseline transcripts. Returns the derived rdr matrices alongside the completed 2D mask.
    """
    logger.info("Determining segment-specific normal spots...")
    n_bins, n_spots = single_X_rdr.shape

    # 1. Override Check: Handle user-provided masks first
    if config.preprocessing.normalidx_file is not None:
        if config.preprocessing.tumorprop_file is not None:
            logger.warning(
                f"Found mixed sources for normal spot definition, assuming {config.preprocessing.normalidx_file}."
            )
        # Standard fallback to whatever downstream logic previously caught 'None'
        return None, None, None, None

    elif single_tumor_prop is not None:
        logger.info("Identifying normal spots based on provided tumor proportion.")
        global_mask = np.zeros(n_spots, dtype=np.bool_)
        for prop_threshold in np.arange(0.05, 0.6, 0.05):
            global_mask = single_tumor_prop < prop_threshold
            if np.sum(single_X_rdr[:, global_mask]) > 200 * single_X.shape[0]:
                logger.info(
                    f"Determined normal spots with sufficient UMIs based on input tumor proportion @ prop_threshold={prop_threshold}"
                )
                break
        else:
            logger.warning(
                "Failed to determine normal spots with sufficient UMIs based on input tumor proportion."
            )

        # Broadcast user's 1-dimensional array into 2-dimensional uniform mask
        local_normal_mask = np.tile(global_mask, (n_bins, 1))

    # 2. Dynamic Discovery: Run Numba optimized sliding-window
    else:
        logger.info(
            "Determining normal mask dynamically based on intra/inter-cluster statistical bounds."
        )
        local_normal_mask, avg_clones_included = compute_local_normal_mask(
            baf_profiles=baf_profiles,
            spot_assignments=np.array(res["new_assignment"], dtype=np.int32),
            single_X_rdr=single_X_rdr,
            window_size=window_size,
        )
        logger.info(
            f"Average effective number of clones included per segment: {avg_clones_included:.2f}"
        )

    # 3. Base Compute execution
    mean_sparsity = np.mean(np.sum(local_normal_mask, axis=1) == 0)
    logger.info(
        f"Found sparsity of segment-normal spot set = {100. * mean_sparsity:.3f}%"
    )

    min_norm = config.quality.min_normal_count_perbin
    rdr_normal, spots_coverage = _compute_baseline_core(
        single_X_rdr, local_normal_mask, min_norm
    )

    # Calculate normalization outer-product and report
    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)
    confident_fraction = np.mean(rdr_normal > 0)

    logger.info(
        f"Found {100. * confident_fraction:.3f}% of segments with confident normal baseline for "
        f"MIN_NORMAL_COUNT_PERBIN={min_norm}"
    )

    return rdr_normal, single_X_rdr, single_base_nb_mean, local_normal_mask


'''
def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2, # LogFC threshold for dropping a gene between Unsure and Normal spots.
    logfcthreshold_t=4, # LogFC threshold for dropping a gene between the Tumor and Normal spots
    quantile_threshold=80, # percentile of total UMI counts a gene must exceed to be considered for dropping.
):
    """
    Cluster input transcripts per slice into "normal" vs "tumor" spots based on pca + kmeans,
    utilizing pre-labeled "normal" candidates to identify the "normal" cluster.

    Drop gene transcripts that are differentially expressed between this "normal" cluster
˚   and the "tumor" spots based on log fold change.

    Returns new counts structure of (genomic bins x spots) after filtering genes with estimated
    differential expression, namely new_single_X_rdr.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}

    # NB gene_umis summed over spots.
    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    # NB map of unique integer per gene.
    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    # NB loop over slices.
    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]

        # NB adata for this slice.
        tmpadata = adata[index, :].copy()

        # NB insufficient normal spot umis for this slice.
        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10  # MAGIC
        ):
            logger.warning(f"TODO!")
            continue

        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )

        # NB  filter genes based on number of cells or counts.
        #     see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.filter_genes.html
        sc.pp.filter_genes(tmpadata, min_cells=10)

        # NB median number of umis per spot?
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))

        # NB normalize such that every spot has the same total count after normalization.
        #    see https://scanpy.readthedocs.io/en/1.9.x/generated/scanpy.pp.normalize_total.html
        sc.pp.normalize_total(tmpadata, target_sum=med)

        # NB log(1 + x) transform.
        #    see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.log1p.html
        sc.pp.log1p(tmpadata)

        # NB adds PCA representation of data: adata.obsm['X_pca' with shape (adata.n_obs, n_comps)
        #    see https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html
        sc.pp.pca(tmpadata, n_comps=4)

        # NB fit two clusters to PCA representation of data.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
        kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])

        # NB determine which cluster corresponds to normal candidates.
        idx_kmeans_label = np.argmax(
            np.bincount(kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2)
        )

        # NB all normal candidates are "normal".
        clone = np.array(["normal"] * tmpadata.shape[0])
        clone[
            (kmeans_labels != idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "tumor"

        # NB spots that are in the same kmeans cluster as normal candidates but not pre-labeled as "normal candidates"
        clone[
            (kmeans_labels == idx_kmeans_label) & (~tmpadata.obs["normal_candidate"])
        ] = "unsure"

        tmpadata.obs["clone"] = clone

        # NB aggregate counts per normal/tumor designation.
        agg_counts = np.vstack(
            [
                np.sum(
                    tmpadata.layers["count"][tmpadata.obs["clone"] == label, :], axis=0
                )
                for label in ["normal", "unsure", "tumor"]
            ]
        )
        agg_counts = agg_counts / np.sum(agg_counts, axis=1, keepdims=True) * 1e6

        # NB total umis per gene for genes corresponding to adata.var.index
        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])

        # TODO divide-by-zero errors >>>>
        # NB log fold change normal vs unsure
        logfc_u = np.where(
            ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[1, :] / agg_counts[0, :]),
        )

        # NB log fold change normal vs tumor
        logfc_t = np.where(
            ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
            10,
            np.log2(agg_counts[2, :] / agg_counts[0, :]),
        )
        # <<<<<
        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set

        logger.info(
            f"Removed {len(filtered_out_set)} genes with differential expression based on normal spots."
        )

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    total_counts, retained_counts = 0, 0

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        # RDR (genes)
        bin_genes = set(genestr.split(" "))
        involved_genes = bin_genes - filtered_out_set

        total_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(bin_genes)]
        )
        retained_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)]
        )

        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    logger.info(f"Retained {100. * retained_counts / total_counts:.3f}% of bin UMIs.")

    return new_single_X_rdr, filtered_out_set
'''


def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2,
    logfcthreshold_t=4,
    quantile_threshold=80,
    use_kmeans=True,
):
    """
    Cluster input transcripts per slice into "normal" vs "tumor" spots based on pca + kmeans
    (or directly via pre-defined normal candidates), dropping differentially expressed genes.

    Returns new counts structure of (genomic bins x spots) after filtering genes with estimated
    differential expression, namely new_single_X_rdr.
    """
    adata = anndata.AnnData(exp_counts)
    adata.layers["count"] = exp_counts.values
    adata.obs["normal_candidate"] = normal_candidate

    map_gene_adatavar, map_gene_umi = {}, {}

    list_gene_umi = np.sum(adata.layers["count"], axis=0)

    for i, x in enumerate(adata.var.index):
        map_gene_adatavar[x] = i
        map_gene_umi[x] = list_gene_umi[i]

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    for s, sname in enumerate(sample_list):
        if sname is None:
            index = np.arange(adata.shape[0])
        else:
            index = np.where(sample_ids == s)[0]

        tmpadata = adata[index, :].copy()

        if (
            np.sum(tmpadata.layers["count"][tmpadata.obs["normal_candidate"], :])
            < tmpadata.shape[1] * 10
        ):
            logger.warning(
                f"Insufficient normal spot UMIs for slice {sname}. Skipping."
            )
            continue

        umi_threshold = np.percentile(
            np.sum(tmpadata.layers["count"], axis=0), quantile_threshold
        )

        sc.pp.filter_genes(tmpadata, min_cells=10)
        med = np.median(np.sum(tmpadata.layers["count"], axis=1))
        sc.pp.normalize_total(tmpadata, target_sum=med)
        sc.pp.log1p(tmpadata)

        # ---------------------------------------------------------
        # CLUSTERING LOGIC: K-Means vs Pre-Defined
        # ---------------------------------------------------------
        if use_kmeans:
            sc.pp.pca(tmpadata, n_comps=4)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(tmpadata.obsm["X_pca"])
            kmeans_labels = kmeans.predict(tmpadata.obsm["X_pca"])

            # Log cluster proportions
            n_spots_slice = len(kmeans_labels)
            counts = np.bincount(kmeans_labels, minlength=2)
            logger.info(
                f"K-means clustering for slice {sname}: "
                f"Cluster 0: {100.0 * counts[0] / n_spots_slice:.1f}% | "
                f"Cluster 1: {100.0 * counts[1] / n_spots_slice:.1f}%"
            )

            idx_kmeans_label = np.argmax(
                np.bincount(
                    kmeans_labels[tmpadata.obs["normal_candidate"]], minlength=2
                )
            )

            # Log overlap with normal candidates
            normal_mask = tmpadata.obs["normal_candidate"].values
            n_candidates = np.sum(normal_mask)
            overlap = np.sum(kmeans_labels[normal_mask] == idx_kmeans_label)

            if n_candidates > 0:
                logger.info(
                    f"Overlap: {overlap}/{n_candidates} ({100.0 * overlap / n_candidates:.1f}%) "
                    f"of pre-defined normal candidates fell into the designated 'normal' Cluster {idx_kmeans_label}."
                )

            clone = np.array(["normal"] * tmpadata.shape[0], dtype=object)
            clone[(kmeans_labels != idx_kmeans_label) & (~normal_mask)] = "tumor"
            clone[(kmeans_labels == idx_kmeans_label) & (~normal_mask)] = "unsure"

        else:
            # Bypass K-means entirely, rely on pre-labeled normal candidates
            logger.info(
                f"Skipping K-means for slice {sname}. Using pre-defined normal candidates directly."
            )
            normal_mask = tmpadata.obs["normal_candidate"].values

            clone = np.array(["tumor"] * tmpadata.shape[0], dtype=object)
            clone[normal_mask] = "normal"
            # Note: There are no "unsure" spots in this mode.

        tmpadata.obs["clone"] = clone

        # ---------------------------------------------------------
        # DIFFERENTIAL EXPRESSION CALCULATION
        # ---------------------------------------------------------
        agg_counts = np.vstack(
            [
                np.sum(
                    tmpadata.layers["count"][tmpadata.obs["clone"] == label, :], axis=0
                )
                for label in ["normal", "unsure", "tumor"]
            ]
        )
        # Avoid division by zero on empty slices/clusters
        row_sums = np.sum(agg_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        agg_counts = agg_counts / row_sums * 1e6

        geneumis = np.array([map_gene_umi[x] for x in tmpadata.var.index])

        has_unsure = np.sum(tmpadata.obs["clone"] == "unsure") > 0
        has_tumor = np.sum(tmpadata.obs["clone"] == "tumor") > 0

        # Safe logFC for unsure
        if has_unsure:
            logfc_u = np.where(
                ((agg_counts[1, :] == 0) | (agg_counts[0, :] == 0)),
                10,
                np.log2(agg_counts[1, :] / agg_counts[0, :]),
            )
        else:
            logfc_u = np.zeros(tmpadata.shape[1])

        # Safe logFC for tumor
        if has_tumor:
            logfc_t = np.where(
                ((agg_counts[2, :] == 0) | (agg_counts[0, :] == 0)),
                10,
                np.log2(agg_counts[2, :] / agg_counts[0, :]),
            )
        else:
            logfc_t = np.zeros(tmpadata.shape[1])

        this_filtered_out_set = set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)
                ]
            )
        ) | set(
            list(
                tmpadata.var.index[
                    (np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)
                ]
            )
        )
        filtered_out_set = filtered_out_set | this_filtered_out_set

        logger.info(
            f"Removed {len(this_filtered_out_set)} genes with differential expression for slice {sname}. "
            f"Total unique removed across run: {len(filtered_out_set)}."
        )

    new_single_X_rdr = np.zeros((df_bininfo.shape[0], adata.shape[0]))
    total_counts, retained_counts = 0, 0

    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        bin_genes = set(genestr.split(" "))
        involved_genes = bin_genes - filtered_out_set

        total_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(bin_genes)]
        )
        retained_counts += np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)]
        )

        new_single_X_rdr[b, :] = np.sum(
            adata.layers["count"][:, adata.var.index.isin(involved_genes)], axis=1
        )

    logger.info(
        f"Retained {100. * retained_counts / max(total_counts, 1):.3f}% of bin UMIs."
    )

    return new_single_X_rdr, filtered_out_set


def normal_baf_bin_filter(
    df_gene_snp,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    nu,
    logphase_shift,
    index_normal,
    geneticmap_file,
    confidence_interval=None,
    min_betabinom_tau=30,
):
    """
    Calculates new aggregated counts (genome blocks x spots) after filtering genomic bins
    adjudged to have non-normal-like baf in the __normal clone__.

    This may be the case if
     - the 'normal' clone is erroneously identified;
     - its mixed with non-normal spots
     - there is allele-specific expression.

    Returns:
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        df_gene_snp,
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(
            get_global_config().quality.normal_allele_specific_confidence
        )

    logger.info("Selecting bins for removal based on normal spot BAF.")

    # NB pool b-allele counts for each bin across all normal spots; 1D genomic segments.
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)

    # TODO
    model = Weighted_BetaBinom(
        tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD
    )

    # LEGACY
    settings = get_em_solver_params()
    tmpres = model.fit(**settings)

    logger.info(
        f"Best-fit BetaBinom model to normal spot BAF has parameters={tmpres.params}"
    )

    # TODO warn if patched.
    # NB patches parameters assuming min_betabinom_tau=30;
    tmpres.params[0] = 0.5
    tmpres.params[-1] = max(tmpres.params[-1], min_betabinom_tau)

    # NB remove bins if "normal" b-allele probabilities fall out of (5%-95%) confidence interval,
    #    this may be the case if mixed with non-normal spots or allele-specific expression present.
    #
    removal_indicator1 = tmpX < scipy.stats.betabinom.ppf(
        confidence_interval[0],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1.0 - tmpres.params[0]) * tmpres.params[1],
    )
    removal_indicator2 = tmpX > scipy.stats.betabinom.ppf(
        confidence_interval[1],
        tmptotal_bb_RD,
        tmpres.params[0] * tmpres.params[1],
        (1.0 - tmpres.params[0]) * tmpres.params[1],
    )

    index_removal = np.where(removal_indicator1 | removal_indicator2)[0]
    index_remaining = np.where(~(removal_indicator1 | removal_indicator2))[0]

    removal_indicator = removal_indicator1 | removal_indicator2

    logger.info(
        f"Removing {np.count_nonzero(removal_indicator)} [{100. * np.mean(removal_indicator):.4f}]% genomic segments with potential allele-specific expression, based on normal candidates --- confidence={confidence_interval} and min_betabinom_tau={min_betabinom_tau}."
    )

    # NB below constructs single_X, single_base_nb_mean, single_total_bb_RD with segments removed.
    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None

    # NB reassign bin_id to be unique integers in [0, n_bins_remaining) for downstream processing.
    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map(
        {x: i for i, x in enumerate(index_remaining)}
    )
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    logger.info(f"Solved for unique bin ids:\n{np.unique(df_gene_snp.bin_id)}")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(
            f"Detected {df_gene_snp.bin_id.isnull().sum()} bin_ids with NaN value."
        )

    single_X = single_X[index_remaining, :, :]
    single_base_nb_mean = single_base_nb_mean[index_remaining, :]
    single_total_bb_RD = single_total_bb_RD[index_remaining, :]

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(
            df_gene_snp[
                (df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())
            ].bin_id.unique()
        )

    assert (
        df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0]
    ), f"{df_gene_snp['bin_id'].notna().sum()} != {single_X.shape[0]}"
    assert df_gene_snp["bin_id"].nunique(dropna=True) == sum(
        lengths
    ), f"{df_gene_snp['bin_id'].notna().sum()} != {sum(lengths)}"

    return df_gene_snp, SpatioGenomicCounts(lengths, single_X, single_base_nb_mean, single_total_bb_RD)
