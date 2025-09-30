import logging
import numpy as np
from scipy.special import logsumexp
from numba import njit

logger = logging.getLogger(__name__)


# TODO
# @njit
def icm_update(
    single_llf,
    adjacency_list,
    new_assignment,
    spatial_weight,
    posterior,
    tol=0.01,
    log_persample_weights=None,
    sample_ids=None,
):
    # NB ICM is guranteed to converge.
    n_spots, n_clones = single_llf.shape
    w_edge = np.zeros(n_clones)
    niter = 0

    while True:
        # NB number edits in this sweep.
        edits = 0

        for i in range(n_spots):
            # NB emission likelihood for all clones for this spot
            w_node = single_llf[i, :].copy()

            # NB sample/slice for this spot.
            this_sample = sample_ids[i]

            if log_persample_weights is not None:
                w_node += log_persample_weights[:, this_sample]

            # NB edge costs accumulated across clones
            w_edge[:] = 0.0

            # NB sum spatial weights for neighbors grouped by current assignment
            for j, edge_weight in adjacency_list[i]:
                neighbor_assignment = new_assignment[j]
                w_edge[neighbor_assignment] += edge_weight

            assignment_cost = w_node + spatial_weight * w_edge

            # NB ICM is greedy picking of best clone for each spot.
            label = np.argmax(assignment_cost)

            # logger.info(f"ICM label contention: {assignment_cost} implies {new_assignment[i]} -> {label}")

            edits += int(label != new_assignment[i])
            new_assignment[i] = label

            # TODO
            norm = logsumexp(assignment_cost)
            posterior[i, :] = np.exp(assignment_cost - norm)

        edit_rate = edits / n_spots
        niter += 1

        _, cnts = np.unique(new_assignment, return_counts=True)

        logger.info(f"Found ICM edit_rate={edit_rate:.6f} for iteration {niter}.")
        logger.info(f"Found ICM inferred clone proportions: {cnts / n_spots}")

        if edit_rate < tol:
            break

    return niter
