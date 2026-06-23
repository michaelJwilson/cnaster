import numpy as np
from numba import njit
# from dataclasses import dataclass, asdict, field

# from collections import deque
from cnaster.config import start_time
from cnaster.logger import get_logger
from cnaster.hmrf_utils import hmrf_perf_entry
# from cnaster.wolff import build_wolff_cluster
from collections import deque

logger = get_logger(__name__, start_time=start_time)


def get_clone_split(assignment):
    """
    Return an array of clone proportion given an assignment.
    """
    unique_ids, counts = np.unique(assignment.astype(int), return_counts=True)

    max_id = unique_ids.max()
    size = 1 + max_id

    full_counts = np.zeros(size, dtype=int)
    full_counts[unique_ids] = counts

    return full_counts / full_counts.sum()


def unpack_adjacency(adj_list):
    # TODO? hash map for O(1) lookup?
    adj_spots, adj_neighbors, adj_weights = [], [], []

    # NB spot repeated for each of its neighbors.
    for spot, neighbors in enumerate(adj_list):
        for neighbor, weight in neighbors:
            adj_spots.append(spot)
            adj_neighbors.append(neighbor)
            adj_weights.append(weight)

    adj_spots, adj_neighbors, adj_weights = (
        np.array(adj_spots, dtype=int),
        np.array(adj_neighbors, dtype=int),
        np.array(adj_weights, dtype=float),
    )

    _, cnts = np.unique(adj_spots, return_counts=True)

    logger.debug(f"Found adjaceny neighbors counts={cnts}")

    return adj_spots, adj_neighbors, adj_weights


@njit(cache=True)
def calc_cluster_assignment_cost(
    cluster,
    single_llf,
    sample_ids,
    log_persample_weights,
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    n_clones,
):
    # NB all spots in cluster have the same initial spin.
    w_node = np.zeros(n_clones, dtype=np.float64)
    w_edge = np.zeros(n_clones, dtype=np.float64)
    # n_spots = single_llf.shape[0]

    start_k = 0

    # NB spots in cluster are monotonically increasing.
    for ii in range(len(cluster)):
        spot = cluster[ii]

        # NB likelihoods for each clone, for this spot.
        w_node += single_llf[spot, :]

        # NB expected clone proportions for this slice.
        if log_persample_weights is not None:
            this_sample = sample_ids[spot]
            w_node += log_persample_weights[:, this_sample]

        found = False

        # NB adjacency is symmetric.
        for k in range(start_k, adjacency_spots.shape[0]):
            if adjacency_spots[k] == spot:
                neighbor = adjacency_neighbors[k]
                edge_weight = adjacency_weights[k]

                # NB i and j both in cluster ergo always aligned; we will revisit on j.
                #    convention set by icm_sweep, which double counts edges.
                if neighbor in cluster:
                    w_edge += edge_weight / 2.0

                # NB neighbor not in cluster; only if new cluster assignment
                #    aligns with spin is there a preference; we will not revisit neighbor in this.
                else:
                    neighbor_assignment = new_assignment[neighbor]
                    w_edge[neighbor_assignment] += edge_weight

                found = True
            else:
                if found:
                    # NB we can start here in the neighbor list for the next spot in the cluster,
                    #    as monotonically increasing.
                    start_k = k
                    break

    return w_node, w_edge


@njit
def logsumexp(x):
    x_max = np.max(x)
    s = 0.0
    for i in range(x.shape[0]):
        s += np.exp(x[i] - x_max)
    return x_max + np.log(s)


@njit(cache=True)
def calc_assignment_cost(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    log_persample_weights=None,
    sample_ids=None,
):
    n_spots, _ = single_llf.shape
    cost = 0.0

    for i in range(n_spots):
        spot_assignment = new_assignment[i]
        cost += single_llf[i, spot_assignment]

        # NB log_persample_weights (n_clone, n_sample/n_slice);
        #    exp. proportion of clone per slice.
        if log_persample_weights is not None:
            this_sample = sample_ids[i]
            cost += log_persample_weights[spot_assignment, this_sample]

        mask = adj_spots == i
        neighbors = adj_neighbors[mask]
        weights = adj_weights[mask]

        # NB if the spot assignment agrees with its neighbor, the cost increases.
        for neighbor, edge_weight in zip(neighbors, weights):
            neighbor_assignment = new_assignment[neighbor]

            if neighbor_assignment == spot_assignment:
                cost += spatial_weight * edge_weight / 2.0

    return cost


# TODO
# @njit(cache=True)
def merge_assignment(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    assignment,
    spatial_weight,
    log_persample_weights=None,
    sample_ids=None,
):
    n_spots, n_clones = single_llf.shape

    # NB unary_sum[u, k] stores the sum of likelihoods for label k
    #    for all spots currently assigned to label u.
    unary_sum = np.zeros((n_clones, n_clones), dtype=np.float64)

    # NB boundary_gain[u, v] stores the potential spatial gain if u and v are merged.
    boundary_gain = np.zeros((n_clones, n_clones), dtype=np.float64)

    current_spatial_cost = 0.0

    for i in range(n_spots):
        u = assignment[i]

        # NB accumulate unary terms for this spot across all potential labels.
        for k in range(n_clones):
            val = single_llf[i, k]
            if log_persample_weights is not None:
                val += log_persample_weights[k, sample_ids[i]]

            unary_sum[u, k] += val

        mask = adj_spots == i
        neighbors = adj_neighbors[mask]
        weights = adj_weights[mask]

        for neighbor, edge_weight in zip(neighbors, weights):
            v = assignment[neighbor]

            if u == v:
                current_spatial_cost += spatial_weight * edge_weight / 2.0
            else:
                boundary_gain[u, v] += spatial_weight * edge_weight / 2.0

    current_unary_cost = 0.0

    for c in range(n_clones):
        current_unary_cost += unary_sum[c, c]

    # NB meets validation of cost given by calc_assignment_cost.
    current_total_cost = current_unary_cost + current_spatial_cost

    best_merge_cost = -np.inf
    best_merge_pair = (-1, -1)

    for u in range(n_clones):
        for v in range(n_clones):
            if u == v:
                continue

            if boundary_gain[u, v] > 0:
                # NB Option: Merge u into v (spots of u become v).
                #    Delta = (unary of u becoming v) - (unary of u being u) + boundary gain.
                delta_u_to_v = (unary_sum[u, v] - unary_sum[u, u]) + boundary_gain[u, v]

                if current_total_cost + delta_u_to_v > best_merge_cost:
                    best_merge_cost = current_total_cost + delta_u_to_v
                    best_merge_pair = (u, v)

    if best_merge_cost > -np.inf:
        logger.info(
            f"Found best merge pair {best_merge_pair} with dC={best_merge_cost - current_total_cost:.6e}."
        )
    else:
        logger.info(f"No beneficial merge found among {n_clones} clones.")

    return current_total_cost, best_merge_cost, best_merge_pair


@njit(cache=True)
def icm_sweep(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    tol=0.0,
    log_persample_weights=None,
    sample_ids=None,
    cost_zeropoint=0.0,
    temp=1.0,
    min_clone_spots=200,
    max_iter=5,
):
    # NB ICM is guranteed to converge to a local (maximum).
    n_spots, n_clones = single_llf.shape
    w_edge = np.zeros(n_clones)
    niter = 0

    cost = cost_zeropoint

    # TODO no logger given njit, but warning on iterations exceeded?
    while niter < max_iter:
        # NB number edits in this sweep.
        edits = 0
        clone_counts = np.zeros(n_clones, dtype=np.int32)

        for i in range(n_spots):
            # NB emission likelihood for all clones for this spot; (1, n_clone).
            w_node = single_llf[i, :].copy()

            # NB log_persample_weights (n_clone, n_sample/n_slice);
            #    exp. proportion of clone per slice.
            if log_persample_weights is not None:
                this_sample = sample_ids[i]
                w_node += log_persample_weights[:, this_sample]

            # NB edge costs accumulated across clones: idx represent a clone assignment
            #    for this spot; every neighbor with the same assignment contributes positively
            #    to w_edge[idx].
            w_edge[:] = 0.0

            # NB sum spatial weights for neighbors grouped by current assignment
            # TODO
            mask = adj_spots == i
            neighbors = adj_neighbors[mask]
            weights = adj_weights[mask]

            # NB if the spot assignment agrees with its neighbor, the cost increases.
            for neighbor, edge_weight in zip(neighbors, weights):
                neighbor_assignment = new_assignment[neighbor]
                w_edge[neighbor_assignment] += edge_weight

            # NB assignment cost to each clone for this spot.
            assignment_cost = w_node + (spatial_weight / temp) * w_edge

            # NB ICM is greedy picking of best clone with maximum likelihood for each spot.
            label = np.argmax(assignment_cost)

            # NB may double count if a spot label changes repeatedly.
            edits += int(label != new_assignment[i])
            cost += assignment_cost[label] - assignment_cost[new_assignment[i]]

            new_assignment[i] = label
            clone_counts[new_assignment[i]] += 1

            # TODO
            norm = logsumexp(assignment_cost)
            posterior[i, :] = np.exp(assignment_cost - norm)

        edit_rate = edits / n_spots

        if min_clone_spots > 0 and clone_counts.min() < min_clone_spots:
            eligible = np.where(clone_counts >= min_clone_spots)[0]

            for c in range(n_clones):
                if (
                    len(eligible) > 0
                    and clone_counts[c] < min_clone_spots
                    and clone_counts[c] > 0
                ):
                    spot_indices = np.where(new_assignment == c)[0]
                    new_labels = eligible[
                        np.random.randint(0, len(eligible), size=len(spot_indices))
                    ]

                    for idx, new_label in zip(spot_indices, new_labels):
                        new_assignment[idx] = new_label
                        clone_counts[c] -= 1
                        clone_counts[new_label] += 1

            edit_rate = np.inf

        niter += 1

        if edit_rate <= tol:
            break

    return niter, cost


def icm_sweep_deque(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    tol=0.0,
    log_persample_weights=None,
    sample_ids=None,
    cost_zeropoint=0.0,
    temp=1.0,
    min_clone_spots=200,
):
    n_spots, n_clones = single_llf.shape
    w_edge = np.zeros(n_clones)
    niter = 0
    cost = cost_zeropoint

    queue = deque(range(n_spots))
    in_queue = np.ones(n_spots, dtype=bool)

    clone_counts = np.zeros(n_clones, dtype=np.int32)
    for idx in range(n_spots):
        clone_counts[new_assignment[idx]] += 1

    logger.info(
        f"Starting icm sweep with clone proportion:\n{clone_counts / clone_counts.sum()}"
    )

    # TODO
    min_spot_guard = 0

    while queue:
        edits = 0

        # NB future batches populated with the neighbors of edits in this batch.
        for _ in range(len(queue)):
            i = queue.popleft()
            in_queue[i] = False

            w_node = single_llf[i, :].copy()
            if log_persample_weights is not None:
                this_sample = sample_ids[i]
                w_node += log_persample_weights[:, this_sample]
            w_edge[:] = 0.0

            mask = adj_spots == i
            neighbors = adj_neighbors[mask]
            weights = adj_weights[mask]

            for neighbor, edge_weight in zip(neighbors, weights):
                neighbor_assignment = new_assignment[neighbor]
                w_edge[neighbor_assignment] += edge_weight

            assignment_cost = w_node + (spatial_weight / temp) * w_edge
            label = np.argmax(assignment_cost)

            if label != new_assignment[i]:
                edits += 1
                cost += assignment_cost[label] - assignment_cost[new_assignment[i]]

                clone_counts[new_assignment[i]] -= 1
                clone_counts[label] += 1

                new_assignment[i] = label

                for neighbor in neighbors:
                    if not in_queue[neighbor]:
                        queue.append(neighbor)
                        in_queue[neighbor] = True

            norm = logsumexp(assignment_cost)
            posterior[i, :] = np.exp(assignment_cost - norm)

        batch_edit_rate = edits / n_spots

        logger.info(
            f"Completed icm sweep batch with batch edit rate={batch_edit_rate:.6e}."
        )

        # NB rdr-refinement guard for small baf-identified clones.
        if (
            (min_clone_spots > 0)
            and clone_counts.min() < min_clone_spots
            and clone_counts.min() > 0
        ):
            eligible = np.where(clone_counts >= min_clone_spots)[0]
            for c in range(n_clones):
                if (
                    len(eligible) > 0
                    and clone_counts[c] < min_clone_spots
                    and clone_counts[c] > 0
                ):
                    spot_indices = np.where(new_assignment == c)[0]
                    new_labels = eligible[
                        np.random.randint(0, len(eligible), size=len(spot_indices))
                    ]
                    for idx, new_label in zip(spot_indices, new_labels):
                        new_assignment[idx] = new_label

                        clone_counts[c] -= 1
                        clone_counts[new_label] += 1

            logger.info(
                f"For enforcing min_clone_spot={min_clone_spots} with n_spots={n_spots}, found {len(eligible)} valid clone for reassignment & new clone proportion:\n{clone_counts / clone_counts.sum()}"
            )

            # NB random assignmnent of small clones; force another iteration to reassign.
            if len(eligible) > 1:
                batch_edit_rate = np.inf
                min_spot_guard += 1

        niter += 1

        # NB stop if no edits or only one clone remains.
        if (
            (batch_edit_rate <= tol)
            or np.count_nonzero(clone_counts) <= 1
            or min_spot_guard > 10
        ):
            break

    return niter, cost
