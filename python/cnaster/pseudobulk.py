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
                
    logger.info(f"Merged single_X to pseudobulk of shape {X.shape[2]}.")

    return X, base_nb_mean, total_bb_RD, tumor_prop
