import logging
import numpy as np
from scipy.special import logsumexp
from numba import njit

logger = logging.getLogger(__name__)

def wolff_update(
    single_llf,
    adjacency_list,
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
):
    # TODO p_add should be determined by spatial_weight!
    n_spots, n_clones = single_llf.shape

    # NB pick a spot at random
    this_spot = np.random.randint(n_spots)
    current_assignment = new_assignment[this_spot]

    # NB construct a cluster around this spot of all neighbors with the
    #    same assignment; and them to the cluster with probability P_add
    #    and further add the neighbors of these spots with the same spin
    #    and probability; use a queue.
    cluster, queue = [this_spot], [this_spot]

    # NB 
    p_add = np.random.rand()
    
    while queue:
        current = queue.pop(0)

        for neighbor, edge_weight in adjacency_list[current]:
            if new_assignment[neighbor] == current_assignment:
                if (neighbor not in cluster) and np.random.rand() < p_add:
                    cluster.append(neighbor)
                    queue.append(neighbor)

    logger.info(f"Solved for a cluster of {len(cluster)} spins with p_add={p_add}")
                    
    w_node, w_edge = np.zeros(n_clones, dtype=float), np.zeros(n_clones, dtype=float)

    for spot in cluster:
        w_node += single_llf[spot, :]

        this_sample = sample_ids[spot]

        if log_persample_weights is not None:
            w_node += log_persample_weights[:, this_sample]

        # TODO do not double count edges.
        for neighbor, edge_weight in adjacency_list[spot]:
            # NB i and j both in cluster; we will revisit on j.
            if neighbor in cluster:
                w_edge += edge_weight / 2.

            # NB neighbor not in cluster; only if new cluster assignment
            #    aligns with spin is there a preference; we will not revisit neighbor in this.
            else:
                neighbor_assignment = new_assignment[neighbor]
                w_edge[neighbor_assignment] += edge_weight

    # NB assignment cost to each clone for this cluster.
    assignment_cost = w_node + spatial_weight * w_edge
    current_cost = assignment_cost[current_assignment]

    logger.info(f"Solved for current cost {current_cost:.6e} and new costs=\n{assignment_cost}")
    
    # TODO check.
    assignment_cost[current_assignment] = -np.inf

    # NB Metropolis: if any assignment is lower, accept one randomly.
    #    otherwise, accept with exponential suppression, exp(-delta_cost).
    #
    # NB ignore "cost", we're solving for max.
    best_new_assignment = np.argmax(assignment_cost)
    delta_cost = assignment_cost[best_new_assignment] - current_cost

    acceptance = np.exp(delta_cost)
    
    # NB we always accept the better state (max.)
    if delta_cost > 0:        
        new_cluster_assignment = best_new_assignment
        new_cost = assignment_cost[best_new_assignment]
        
    # NB all proposed states are worse; pick one randomly;
    else:
        if np.random.rand() < np.exp(delta_cost):
            new_cluster_assignment = best_new_assignment
            new_cost = assignment_cost[best_new_assignment]
        else:
            new_cluster_assignment = current_assignment
            new_cost = current_cost
            
    logger.info(f"Solved for better={delta_cost>0} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {current_cost} -> {new_cost} @ acceptance={acceptance:.6e}")

    # TODO define edits.
    for spin in cluster:
        new_assignment[spin] = new_cluster_assignment

    return new_cost


def wolff_sweep(
    single_llf,
    adjacency_list,
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
    p_add=0.5,
    max_iter=100,
):
    # TODO p_add should be determined by spatial_weight!
    n_spots, n_clones = single_llf.shape
    niter, best_cost = 0, -np.inf

    logger.info(f"Solving for a Wolff sweep.")
    
    for i in range(max_iter):
        new_cost = wolff_update(
            single_llf,
            adjacency_list,
            new_assignment,
            spatial_weight,
            posterior,
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
            p_add=p_add
        )

        if new_cost > best_cost:
            best_cost = new_cost
            best_assignment = new_assignment.copy()

            logger.info(f"Found a new best assignment with cost={best_cost:.6e}")
            
    # TODO polish with ICM.
    new_assignment = best_assignment.copy()
    
    return max_iter


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
            # NB emission likelihood for all clones for this spot; (1, n_clone).
            w_node = single_llf[i, :].copy()

            # NB sample/slice for this spot.
            this_sample = sample_ids[i]

            # NB log_persample_weights (n_clone, n_sample/n_slice); 
            #    exp. proportion of clone per slice.
            if log_persample_weights is not None:
                w_node += log_persample_weights[:, this_sample]

            # NB edge costs accumulated across clones
            w_edge[:] = 0.0

            # NB sum spatial weights for neighbors grouped by current assignment
            for j, edge_weight in adjacency_list[i]:
                neighbor_assignment = new_assignment[j]
                w_edge[neighbor_assignment] += edge_weight

            # NB assignment cost to each clone for this spot.
            assignment_cost = w_node + spatial_weight * w_edge

            # NB ICM is greedy picking of best (maximum!) clone for each spot.
            label = np.argmax(assignment_cost)

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
