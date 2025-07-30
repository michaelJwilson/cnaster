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
):
    n_obs = single_X.shape[0]
    n_spots = len(clone_index)

    X = np.zeros((n_obs, 2, n_spots))

    base_nb_mean = np.zeros((n_obs, n_spots))
    total_bb_RD = np.zeros((n_obs, n_spots))

    tumor_prop = np.zeros(n_spots) if single_tumor_prop is not None else None

    for k, idx in enumerate(clone_index):
        if len(idx) == 0:
            logger.warning(f"Clone {k} has no cells, skipping")
            continue

        if single_tumor_prop is not None:
            tumor_mask = single_tumor_prop[idx] > threshold

            idx = idx[tumor_mask]
            tumor_prop[k] = np.mean(single_tumor_prop[idx]) if len(idx) > 0 else 0.0

        # TODO assumes simple aggregation.
        X[:, :, k] = np.sum(single_X[:, :, idx], axis=2)

        base_nb_mean[:, k] = np.sum(single_base_nb_mean[:, idx], axis=1)
        total_bb_RD[:, k] = np.sum(single_total_bb_RD[:, idx], axis=1)

    return X, base_nb_mean, total_bb_RD, tumor_prop
