import logging
import numpy as np
import csv
import time
from pathlib import Path

# from scipy.special import logsumexp
from numba import njit
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class hmrf_perf_entry:
    optimizer: str
    cost: float
    best_cost: float
    padd: float = np.nan
    iteration: int = 0
    ncluster: int = 1
    nedit: int = 0
    clone_split: np.ndarray = field(default_factory=lambda: np.array([-1]))

    def as_dict(self):
        d = asdict(self)
        d["cost"] = "{:+.6e}".format(self.cost)
        d["best_cost"] = "{:+.6e}".format(self.best_cost)
        d["padd"] = "{:.2f}".format(self.padd)
        d["iteration"] = str(self.iteration)
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
    _, cnts = np.unique(assignment, return_counts=True)
    return cnts / len(assignment)


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

    logger.info(f"Found adjaceny neighbors counts={cnts}")

    return adj_spots, adj_neighbors, adj_weights


@njit(cache=True)
def build_wolff_cluster(
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    this_spot,
    p_add,
):
    """
    Construct a cluster around this spot of all neighbors with the
    same assignment; add them to the cluster with probability P_add
    and further add the neighbors of these spots with the same spin
    and probability; use a queue.
    """
    visited, cluster, queue = [this_spot], [this_spot], [this_spot]
    current_assignment = new_assignment[this_spot]

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

            # DEPRECATE a new neighbor.
            if neighbor not in cluster and (
                new_assignment[neighbor] == current_assignment
            ):
                if np.random.rand() < p_add:
                    cluster.append(neighbor)
                    queue.append(neighbor)

    return np.array(sorted(list(cluster)))


@njit(cache=True)
def calc_assignment_cost(
    cluster,
    single_llf,
    sample_ids,
    log_persample_weights,
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    n_clones,
    spatial_weight,
):
    w_node = np.zeros(n_clones, dtype=np.float64)
    w_edge = np.zeros(n_clones, dtype=np.float64)
    n_spots = single_llf.shape[0]

    start_k = 0
    
    # NB spots in cluster are monotonically increasing.
    for ii in range(len(cluster)):
        spot = cluster[ii]
        spot_assignment == new_assignment[spot]

        # NB likelihoods for each clone, for this spot.
        w_node += single_llf[spot, :]

        this_sample = sample_ids[spot]

        # NB expected clone proportions for this slice.
        if log_persample_weights is not None:
            w_node += log_persample_weights[:, this_sample]

        found = False

        # NB adjacency is symmetric.
        for k in range(start_k, adjacency_spots.shape[0]):
            if adjacency_spots[k] == spot:
                neighbor = adjacency_neighbors[k]
                edge_weight = adjacency_weights[k]

                # NB i and j both in cluster; we will revisit on j.
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
                    # NB we can start here in the neighbor list for the next spot in the cluster.
                    start_k = k
                    break

    return w_node, w_edge


def wolff_update(
    single_llf,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
    p_add=0.0,
    cost_zeropoint=0.0,
    temp=None,
):
    # TODO p_add should be determined by spatial_weight!
    n_spots, n_clones = single_llf.shape

    # NB pick a spot at random
    # TODO smarter choice?
    this_spot = np.random.randint(n_spots)
    current_assignment = new_assignment[this_spot]

    cluster = build_wolff_cluster(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        p_add,
    )

    # NB equivalent to (locally) optimal ICM; return original cost and null op. cluster.
    if len(cluster) <= 1:
        return cost_zeropoint, current_assignment, cluster

    # NB relative cost for assignment to each clone for posed cluster.
    node_cost, edge_cost = calc_assignment_cost(
        cluster,
        single_llf,
        sample_ids,
        log_persample_weights,
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        n_clones,
        spatial_weight,
    )

    # NB assignment cost to each clone for this cluster.
    assignment_cost = node_cost + spatial_weight * edge_cost

    logger.info(f"Found node and edge costs:\n{node_cost}\n{edge_cost}")

    current_cost = assignment_cost[current_assignment]

    # NB we look for the next best ...
    assignment_cost[current_assignment] = -np.inf

    # NB Metropolis: if any assignment is lower, accept one randomly.
    #    otherwise, accept with exponential suppression, exp(-delta_cost).
    #
    # NB ignore "cost", we're solving for max.
    best_new_assignment = np.argmax(assignment_cost)
    delta_cost = assignment_cost[best_new_assignment] - current_cost

    # NB all proposed states are worse; pick one randomly;
    accepted = (
        np.random.rand() < np.exp(delta_cost / temp) if temp is not None else False
    )

    logger.info(
        f"Solved for a cluster of {len(cluster):4d} spins @ p_add={p_add:.3f} with current cost {current_cost:.4e}, next best cost={assignment_cost[best_new_assignment]:.4e} and dE={delta_cost:.4e}; accepted={accepted}."
    )

    # NB we always accept the better state (max.), a sampled state, or return the original.
    if delta_cost > 0 or accepted:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment
    else:
        new_cluster_assignment, new_cost, cluster = (
            current_assignment,
            cost_zeropoint,
            None,
        )

    """
    logger.info(
        f"Solved for better={int(delta_cost>0)} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {cost_zeropoint} -> {new_cost} @ min_acceptance={min_acceptance}"
    )
    """

    return new_cost, new_cluster_assignment, cluster


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
    max_iter=250,
    cost_zeropoint=0.0,
):
    logger.info(
        f"Completing an ICM sweep for unary likelihood of shape {single_llf.shape} and spatial weight {spatial_weight}."
    )

    original_assignment = new_assignment.copy()

    # NB icm_sweep updates new_assignment in place.
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
    )

    hmrf_perf_entry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        padd=0.0,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        clone_split=get_clone_split(new_assignment),
    ).log()

    logger.info(f"Found a new best assignment with new cost={new_cost:.6e}")
    logger.info(f"Completing a Wolff sweep.")

    # NB unpacks adjaceny_list into arrays processble by numba.
    best_assignment, best_cost = new_assignment.copy(), new_cost

    for iteration, temp in enumerate(np.logspace(4.0, 0.0, num=max_iter)):
        # TODO tie p_add to temp.
        for p_add in np.arange(0.35, 0.1, -0.05):
            new_cost, new_cluster_assignment, new_cluster = wolff_update(
                single_llf,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                posterior,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
                p_add=p_add,
                cost_zeropoint=new_cost,
                temp=temp,
            )

            # NB when sampling, we always accept the "new" cluster and use the
            #    appropriate zeropoint.
            new_assignment[new_cluster] = new_cluster_assignment

            if new_cost > best_cost:
                best_cost, best_assignment = new_cost, new_assignment.copy()
            
            hmrf_perf_entry(
                optimizer="wolff",
                cost=new_cost,
                best_cost=best_cost,
                padd=p_add,
                iteration=iteration,
                ncluster=len(new_cluster) if new_cluster is not None else 0,
                nedit=np.count_nonzero(new_assignment != original_assignment),
                clone_split=get_clone_split(new_assignment),
            ).log()

            """
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
                cost_zeropoint=new_cost,
            )

            hmrf_perf_entry(
                optimizer="icm",
                cost=new_cost,
                best_cost=best_cost,
                padd=np.nan,
                iteration=iteration,
                nedit=np.count_nonzero(new_assignment != original_assignment),
                clone_split=get_clone_split(new_assignment),
            ).log()
            """

    # NB re-assign with the best found assignment.
    new_assignment[:] = best_assignment

    return max_iter, best_cost


@njit
def logsumexp(x):
    x_max = np.max(x)
    s = 0.0
    for i in range(x.shape[0]):
        s += np.exp(x[i] - x_max)
    return x_max + np.log(s)


@njit(cache=True)
def icm_sweep(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    tol=0.01,
    log_persample_weights=None,
    sample_ids=None,
    cost_zeropoint=0.0,
):
    """
    single_llf: log emission likelihood, to be maximized.
    adj_spots, adj_neighbors, adj_weights: ordered neighbor and weight for all spots.
    new_assignment: array for new assignment, updated in place.
    """
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

            # NB sample/slice for this spot.
            this_sample = sample_ids[i]

            # NB log_persample_weights (n_clone, n_sample/n_slice);
            #    exp. proportion of clone per slice.
            if log_persample_weights is not None:
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

            for neighbor, edge_weight in zip(neighbors, weights):
                neighbor_assignment = new_assignment[neighbor]
                w_edge[neighbor_assignment] += edge_weight

            # NB assignment cost to each clone for this spot.
            assignment_cost = w_node + spatial_weight * w_edge

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

        if edit_rate < tol:
            break

    return niter, cost
