import logging
import numpy as np
import csv
import time
from pathlib import Path
from numba import njit
from dataclasses import dataclass, asdict, field
from collections import deque
from statistics import mean

logger = logging.getLogger(__name__)


@dataclass
class hmrf_perf_entry:
    optimizer: str
    cost: float
    best_cost: float
    iteration: int = 0
    temp: float = np.nan
    acceptance: float = 1.0
    ncluster: int = 1
    nedit: int = 0
    clone_split: np.ndarray = field(default_factory=lambda: np.array([-1]))

    def as_dict(self):
        d = asdict(self)
        d["optimizer"] = d["optimizer"].ljust(15)
        d["cost"] = "{:+.6e}".format(self.cost)
        d["best_cost"] = "{:+.6e}".format(self.best_cost)
        d["iteration"] = str(self.iteration)
        d["temp"] = (
            "Inf".ljust(10) if np.isinf(self.temp) else "{:.4e}".format(self.temp)
        )
        d["acceptance"] = "{:.4e}".format(self.acceptance)
        d["ncluster"] = str(self.ncluster)
        d["nedit"] = "{:d}".format(self.nedit)
        d["clone_split"] = ",".join("{:.8f}".format(x) for x in self.clone_split)

        return d

    def log(self, filename="cnaster_hmrf.perf"):
        perf_dict = self.as_dict()
        perf_file = Path(filename)

        perf_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fieldnames = ["timestamp"] + [k for k in perf_dict.keys() if k != "timestamp"]

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

            if not perf_file.exists():
                writer.writeheader()

            writer.writerow(perf_dict)


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
def build_wolff_cluster(
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    this_spot,
    temp=1.0,
):
    """
    Construct a cluster around this spot of all neighbors with the
    same assignment; add them to the cluster with probability P_add
    and further add the neighbors of these spots with the same spin
    and probability; use a queue.
    """
    visited, cluster, queue = [this_spot], [this_spot], [this_spot]
    current_assignment = new_assignment[this_spot]

    lnprob_forward = 0.0

    while queue:
        current = queue.pop(0)

        mask = adjacency_spots == current
        neighbors = adjacency_neighbors[mask]
        weights = adjacency_weights[mask]

        for neighbor, edge_weight in zip(neighbors, weights):
            # NB we have considered this neighbor already - an effort
            #    to build smaller clusters with less cost.
            if neighbor in visited:
                continue
            else:
                visited.append(neighbor)

            # DEPRECATE a new neighbor by construction.
            if neighbor not in cluster and (
                new_assignment[neighbor] == current_assignment
            ):
                p_add = 1.0 - np.exp(-edge_weight / temp)

                if np.random.rand() <= p_add:
                    cluster.append(neighbor)
                    queue.append(neighbor)
                else:
                    lnprob_forward += np.log(1.0 - p_add)

    return np.array(sorted(list(cluster))), lnprob_forward


@njit(cache=True)
def get_cluster_lnprob_backward(
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    this_spot,
    cluster,
    cluster_assignment,
    temp=1.0,
):
    visited, queue = [this_spot], [this_spot]
    lnprob_backward = 0.0

    while queue:
        current = queue.pop(0)

        mask = adjacency_spots == current
        neighbors = adjacency_neighbors[mask]
        weights = adjacency_weights[mask]

        for neighbor, edge_weight in zip(neighbors, weights):
            # NB we have considered this neighbor already - an effort
            #    to build smaller clusters with less cost.
            if neighbor in visited:
                continue
            else:
                visited.append(neighbor)

            # NB propagate wave front.
            if neighbor in cluster:
                queue.append(neighbor)
            elif new_assignment[neighbor] == cluster_assignment:
                p_add = 1.0 - np.exp(-edge_weight / temp)
                lnprob_backward += np.log(1.0 - p_add)

    return lnprob_backward


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
    n_spots = single_llf.shape[0]

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


@njit(cache=True)
def wolff_update(
    single_llf,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    new_assignment,
    spatial_weight,
    log_persample_weights=None,
    sample_ids=None,
    cost_zeropoint=0.0,
    wolff_temp=None,
    anneal_temp=None,
):
    n_spots, n_clones = single_llf.shape

    # TODO smarter choice?
    this_spot = np.random.randint(n_spots)
    current_assignment = new_assignment[this_spot]

    # NB at high temp. returns a single spin by construction; at low temp. will return max.
    #    clique with shared spin.
    cluster, lnprob_forward = build_wolff_cluster(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        wolff_temp,
    )
    """
    # NB equivalent to (locally) optimal ICM; return original cost and null op. cluster.
    if len(cluster) <= 1:
        # logger.info(f"Solved for a cluster of a single spin (equivalent to ICM).")
        return cost_zeropoint, current_assignment, None
    """
    # NB relative cost for assignment to each clone for posed cluster.
    node_cost, edge_cost = calc_cluster_assignment_cost(
        cluster,
        single_llf,
        sample_ids,
        log_persample_weights,
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        n_clones,
    )

    # NB assignment cost to each clone for this cluster.
    assignment_cost = node_cost + spatial_weight * edge_cost

    # logger.info(f"Found node and edge costs:\n{node_cost}\n{edge_cost}")

    current_cost = assignment_cost[current_assignment]

    # NB we look for the next best ...
    assignment_cost[current_assignment] = -np.inf

    # NB Metropolis: if any assignment is lower, accept one randomly.
    #    otherwise, accept with exponential suppression, exp(-delta_cost).
    #
    # NB ignore "cost", we're solving for max.
    best_new_assignment = np.argmax(assignment_cost)
    delta_cost = assignment_cost[best_new_assignment] - current_cost

    # NB Metropolis step
    if delta_cost > 0:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment
        acceptance = 1.0

        return new_cost, new_cluster_assignment, cluster, acceptance
    """
    # NB sampled / exploration.
    lnprob_backward = get_cluster_lnprob_backward(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        cluster,
        best_new_assignment,
        temp=temp,
    )
    """
    ln_acceptance = delta_cost / anneal_temp
    # ln_acceptance -= lnprob_forward
    # ln_acceptance += lnprob_forward

    acceptance = np.exp(ln_acceptance)
    accepted = np.random.rand() < acceptance
    """
    logger.info(
        f"Solved for a cluster of {len(cluster):4d}/{n_spots:4d} spins @ p_add={p_add:.3f} with current cost {current_cost:.4e},\
        next best cost={assignment_cost[best_new_assignment]:.4e} and dE={delta_cost:.4e}; accepted={accepted}."
    )
    """
    """                                                                                                                                                                                                               
    logger.info(                                                                                                                                                                                                       
        f"Solved for better={int(delta_cost>0)} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {cost_zeropoint} -> {new_cost}"                                                        
    )                                                                                                                                                                                                                 
    """

    # NB we always accept the better state (max.), a sampled state, or return the original.
    if accepted:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment

        return new_cost, new_cluster_assignment, cluster, acceptance
    else:
        return cost_zeropoint, current_assignment, cluster, acceptance


def wolff_sweep(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
):
    """
    logger.info(
        f"Completing an ICM sweep for unary likelihood of shape {single_llf.shape} and spatial weight {spatial_weight}."
    )
    """
    original_assignment = new_assignment.copy()

    cost_zeropoint = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
    )

    hmrf_perf_entry(
        optimizer="zeropoint",
        cost=cost_zeropoint,
        best_cost=cost_zeropoint,
        temp=1.0,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    logger.info(f"Found an initial Potts cost={cost_zeropoint:.6e}.")

    # NB icm_sweep updates new_assignment in place; global max for T=np.inf (independent spins).
    _, new_cost = icm_sweep(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        cost_zeropoint=cost_zeropoint,
        temp=1.0,  # NB ICM is exact for independent spots, "high temperature".
    )

    hmrf_perf_entry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        temp=1.0,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    logger.info(
        f"Found a new best assignment with annealed ICM and new cost={new_cost:.6e}"
    )

    no_edge_cost = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        0.0,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
    )

    hmrf_perf_entry(
        optimizer="no_edge",
        cost=no_edge_cost,
        best_cost=new_cost,
        temp=np.inf,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    # NB unpacks adjaceny_list into arrays processble by numba.
    # TODO BUG? assumes ground state cost is higher than zero icm.
    best_assignment, best_cost = new_assignment.copy(), new_cost
    cost_zeropoint = best_cost

    high_temp = spatial_weight * adj_weights.max()

    # NB base 10 by default!
    # MAGIC HARDCODE
    anneal_temps = np.logspace(-7.0, 1.0 + np.log10(high_temp), num=5_000)[::-1]

    logger.info(
        f"Completing an annealed Wolff sweep with edge range=({adj_weights.min():.4f},{adj_weights.max():.4f}), high temperature {high_temp:.4e}, {len(anneal_temps)} decades:\n{anneal_temps}"
    )

    n_spots = single_llf.shape[0]
    wolff_temp = 4.5

    max_iter = 0

    for anneal_iter, anneal_temp in enumerate(anneal_temps):
        logger.debug(f"Solving for anneal temperature {anneal_temp}.")

        new_assignment = best_assignment.copy()
        cost_zeropoint = best_cost

        for iteration in range(5):
            max_iter += 1

            new_cost, new_cluster_assignment, new_cluster, acceptance = wolff_update(
                single_llf,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
                cost_zeropoint=cost_zeropoint,
                wolff_temp=wolff_temp,  # NB effective cluster size for move.
                anneal_temp=anneal_temp,  # NB annealing of cost.
            )

            if new_cluster is not None:
                cost_zeropoint = new_cost

                for idx in new_cluster:
                    new_assignment[idx] = new_cluster_assignment

            if new_cost > best_cost:
                best_cost, best_assignment = new_cost, new_assignment.copy()

            if anneal_iter % 100 == 0:
                hmrf_perf_entry(
                    optimizer="wolff",
                    cost=new_cost,
                    best_cost=best_cost,
                    iteration=iteration,
                    temp=anneal_temp,
                    acceptance=acceptance,
                    ncluster=len(new_cluster) if new_cluster is not None else 0,
                    nedit=np.count_nonzero(new_assignment != original_assignment),
                    # clone_split=get_clone_split(new_assignment),
                ).log()

    # NB re-assign with the best found assignment.
    new_assignment[:] = best_assignment

    # NB temp=1. by default
    _, new_cost = icm_sweep(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        cost_zeropoint=best_cost,
    )

    hmrf_perf_entry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        iteration=0,
        temp=0.0,
        acceptance=1.0,
        ncluster=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    return max_iter, best_cost


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
@njit(cache=True)
def calc_merge_cost(
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

    return best_merge_cost, best_merge_pair


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
    merge=False,
):
    # NB ICM is guranteed to converge to a local (maximum).
    n_spots, n_clones = single_llf.shape
    w_edge = np.zeros(n_clones)
    niter = 0

    cost = cost_zeropoint

    while True:
        # NB number edits in this sweep.
        edits = 0

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

            # TODO
            norm = logsumexp(assignment_cost)
            posterior[i, :] = np.exp(assignment_cost - norm)

        edit_rate = edits / n_spots
        niter += 1

        # TODO not njit friendly.
        # unique_assignment, cnts = np.unique(new_assignment, return_counts=True)

        # logger.info(f"Found ICM edit_rate={edit_rate:.6f} for iteration {niter}.")
        # logger.info(f"Found ICM inferred clone proportions: {cnts / n_spots}")

        if edit_rate <= tol:
            break

    if merge:
        while True:
            best_merge_cost, best_merge_pair = calc_merge_cost(
                single_llf,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
            )

            if best_merge_cost > cost:
                u, v = best_merge_pair

                for i in range(n_spots):
                    if new_assignment[i] == u:
                        new_assignment[i] = v
                cost = best_merge_cost
            else:
                break
    return niter, cost
