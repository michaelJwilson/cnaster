import logging
import numpy as np

logger = logging.getLogger(__name__)


def merge_pseudobulk_by_index_mix(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    clone_index,
    single_tumor_prop=None,
    threshold=0.5,
    outlier_percentile=None,
):
    n_obs = single_X.shape[0]

    # NB overloads 'spots' as clones.
    n_spots = len(clone_index)

    X = np.zeros((n_obs, 2, n_spots))

    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))

    tumor_prop = np.zeros(n_spots) if single_tumor_prop is not None else None

    if single_tumor_prop is not None:
        logger.warning(
            f"Merging pseudobulk assigning threshold tumor proportion={threshold}"
        )
        
    for k, idx in enumerate(clone_index):
        if len(idx) == 0:
            logger.warning(f"Clone {k} has no cells, skipping")
            continue

        if single_tumor_prop is not None:
            # NB spots in this clone with a given proportion.
            tumor_mask = single_tumor_prop[idx] > threshold

            idx = idx[tumor_mask]

            # NB assumes mean tumor proportion for all spots assigned to this clone.
            tumor_prop[k] = np.mean(single_tumor_prop[idx]) if len(idx) > 0 else 0.0
            
        X[:, :, k] = np.sum(single_X[:, :, idx], axis=-1)
        
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)
        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)

        assert outlier_percentile is None
        
        if outlier_percentile is not None:
            thres = np.percentile(base_nb_mean[:, k], outlier_percentile)

            valid = base_nb_mean[:, k] <= thres
            inlier_norm = 100. * np.sum(base_nb_mean[valid, k]) / outlier_percentile

            outlier_norm = np.sum(base_nb_mean[:, k])

            if outlier_norm > 0.0:
                base_nb_mean[:, k] *= inlier_norm / outlier_norm

                logger.info(f"Applied normal baseline outlier correction={np.mean(inlier_norm / outlier_norm)}")

        percentiles = np.arange(50, 105, 5)

        bafs = X[:, 1, k] / total_bb_RD[:, k]

        valid_rdr = base_nb_mean[:,k] > 0
        rdrs = X[:, 0, k] / base_nb_mean[:,k]

        logger.info(f"Found median BAF={np.median(bafs):.3f} for clone {k}.")
        logger.info(f"Found {len(idx)} spots, mean UMIs per spot={np.sum(X[:, 0, k]) / len(idx):.3f} and mean snp-covering UMIs per spot={np.sum(total_bb_RD[:, k]) / len(idx):.3f} for clone {k}")
        
        if np.any(valid_rdr):
            if not np.isclose(np.nansum(X[:, 0, k]), np.nansum(base_nb_mean[:, k]), rtol=1e-5, atol=1e-6):
                logger.warning(f"Expected consistency between normal baseline normalization total UMI for the clone, {np.nansum(X[:,0,k])} != {np.sum(base_nb_mean[:,k])}")
            
            logger.info(f"Found median RDR={np.median(rdrs[valid_rdr]):.3f} for clone {k} with {100. * np.mean(valid_rdr > 0.0):.3f}% valid.")
            logger.info(f"Found UMI percentiles={np.percentile(X[:, 0, k], percentiles)} for {percentiles} [%].")
            
    logger.info(f"Merged single_X to pseudobulk of shape {X.shape[2]}.")
    
    return X, base_nb_mean, total_bb_RD, tumor_prop
