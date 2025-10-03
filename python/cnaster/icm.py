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
class HMRFPerfEntry:
    optimizer: str
    cost: float
    best_cost: float
    padd: float = np.nan
    iteration: int = 0
    nedit: int = 0
    clone_proportions: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self):
        d = asdict(self)
        d["cost"] = "{:+.6e}".format(self.cost)
        d["best_cost"] = "{:+.6e}".format(self.best_cost)
        d["padd"] = "{:.2f}".format(self.padd) if not np.isnan(self.padd) else ""
        d["iteration"] = str(self.iteration)
        d["nedit"] = nedit
        d["clone_split"] = ",".join(
            "{:.8f}".format(x) for x in self.clone_proportions
        )

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


def get_clone_proportions(assignment):
    _, cnts = np.unique(assignment, return_counts=True)
    return cnts / len(assignment)


def unpack_adjacency(adjacency_list):
    # TODO? hash map for O(1) lookup?
    adjacency_spots, adjacency_neighbors, adjacency_weights = [], [], []

    for spot, neighbors in enumerate(adjacency_list):
        for neighbor, weight in neighbors:
            adjacency_spots.append(spot)
            adjacency_neighbors.append(neighbor)
            adjacency_weights.append(weight)
    return (
        np.array(adjacency_spots, dtype=int),
        np.array(adjacency_neighbors, dtype=int),
        np.array(adjacency_weights, dtype=float),
    )


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
    cluster, queue = set([this_spot]), [this_spot]
    current_assignment = new_assignment[this_spot]

    while queue:
        current = queue.pop(0)

        mask = adjacency_spots == current
        neighbors = adjacency_neighbors[mask]
        weights = adjacency_weights[mask]
        for neighbor, edge_weight in zip(neighbors, weights):
            if neighbor not in cluster and (
                new_assignment[neighbor] == current_assignment
            ):
                if np.random.rand() < p_add:
                    cluster.add(neighbor)
                    queue.append(neighbor)

    return np.array(list(cluster))


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

    cluster_mask = np.zeros(n_spots, dtype=np.uint8)

    for idx in cluster:
        cluster_mask[idx] = 1

    for ii in range(cluster.shape[0]):
        spot = cluster[ii]
        w_node += single_llf[spot, :]

        this_sample = sample_ids[spot]

        if log_persample_weights is not None:
            w_node += log_persample_weights[:, this_sample]

        # TODO adjacency is symmetric?
        # TODO fast forward ...
        for k in range(adjacency_spots.shape[0]):
            if adjacency_spots[k] == spot:
                neighbor = adjacency_neighbors[k]
                edge_weight = adjacency_weights[k]

                # NB i and j both in cluster; we will revisit on j.
                if cluster_mask[neighbor] == 1:
                    w_edge += edge_weight / 2.0
                # NB neighbor not in cluster; only if new cluster assignment
                #    aligns with spin is there a preference; we will not revisit neighbor in this.
                else:
                    neighbor_assignment = new_assignment[neighbor]
                    w_edge[neighbor_assignment] += edge_weight

    # NB assignment cost to each clone for this cluster.
    assignment_cost = w_node + spatial_weight * w_edge

    return assignment_cost


@njit(cache=True)
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
    p_add=0.5,
    cost_zeropoint=0.0,
    min_acceptance=None,
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

    assignment_cost = calc_assignment_cost(
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

    current_cost = assignment_cost[current_assignment]
    """
    logger.info(
        f"Solved for a cluster of {len(cluster)} spins @ p_add={p_add} with current cost {current_cost:.6e} and new costs=\n{assignment_cost}"
    )
    """
    # TODO check.
    assignment_cost[current_assignment] = -np.inf

    # NB Metropolis: if any assignment is lower, accept one randomly.
    #    otherwise, accept with exponential suppression, exp(-delta_cost).
    #
    # NB ignore "cost", we're solving for max.
    best_new_assignment = np.argmax(assignment_cost)
    delta_cost = assignment_cost[best_new_assignment] - current_cost

    new_cluster_assignment, new_cost = current_assignment, cost_zeropoint

    # NB we always accept the better state (max.)
    if delta_cost > 0:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment

        return new_cost, new_cluster_assignment, cluster

    # NB all proposed states are worse; pick one randomly;
    if min_acceptance is not None:
        acceptance = np.exp(delta_cost)

        if np.random.rand() < np.maximum(acceptance, min_acceptance):
            new_cluster_assignment = best_new_assignment
            new_cost = cost_zeropoint + delta_cost

            return new_cost, new_cluster_assignment, cluster
    """
    logger.info(
        f"Solved for better={int(delta_cost>0)} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {cost_zeropoint} -> {new_cost} @ acceptance={acceptance:.6e}"
    )
    """

    return new_cost, new_cluster_assignment, None


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
    max_iter=25,
    cost_zeropoint=0.0,
    min_acceptance=0.5,
):
    logger.info(f"Completing an ICM sweep for unary likelihood of shape {single_llf.shape} and spatial weight {spatial_weight}.")

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

    HMRFPerfEntry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        padd=np.nan,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        clone_proportions=get_clone_proportions(new_assignment),
    ).log()

    logger.info(f"Found a new best assignment with new cost={new_cost:.6e}")
    logger.info(f"Completing a Wolff sweep.")

    # NB unpacks adjaceny_list into arrays processble by numba.
    best_assignment, best_cost = new_assignment.copy(), new_cost

    for p_add in np.arange(0.05, 0.25, 0.05):
        for iteration in range(max_iter):
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
                min_acceptance=min_acceptance,
            )

            # NB when sampling, we always accept the "new" cluster.
            new_assignment[new_cluster] = new_cluster_assignment

            HMRFPerfEntry(
                optimizer="wolff",
                cost=new_cost,
                best_cost=best_cost,
                padd=p_add,
                iteration=iteration,
                nedit=np.count_nonzero(new_assignment != original_assignment),
                clone_proportions=get_clone_proportions(new_assignment),
            ).log()

            if new_cost > best_cost:
                best_cost, best_assignment = new_cost, new_assignment.copy()
                logger.info(f"Found a new best assignment of {len(new_cluster)} spots with new cost={best_cost:.6e}")

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

                if new_cost > best_cost:
                    best_cost, best_assignment = new_cost, new_assignment.copy()
                    logger.info(
                        f"Found a new best assignment with cost={best_cost:.6e} with an additional ICM sweep"
                    )

                HMRFPerfEntry(
                    optimizer="icm",
                    cost=new_cost,
                    best_cost=best_cost,
                    padd=np.nan,
                    iteration=iteration,
                    nedit=np.count_nonzero(new_assignment != original_assignment),
                    clone_proportions=get_clone_proportions(new_assignment),
                ).log()

    new_assignment[:] = best_assignment

    return max_iter


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
    # NB ICM is guranteed to converge.
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

            # NB edge costs accumulated across clones
            w_edge[:] = 0.0

            # NB sum spatial weights for neighbors grouped by current assignment
            mask = adj_spots == i
            neighbors = adj_neighbors[mask]
            weights = adj_weights[mask]

            for neighbor, edge_weight in zip(neighbors, weights):
                neighbor_assignment = new_assignment[neighbor]
                w_edge[neighbor_assignment] += edge_weight

            # NB assignment cost to each clone for this spot.
            assignment_cost = w_node + spatial_weight * w_edge

            # NB ICM is greedy picking of best (maximum!) clone for each spot.
            label = np.argmax(assignment_cost)

            edits += int(label != new_assignment[i])
            cost += assignment_cost[label] - assignment_cost[new_assignment[i]]

            new_assignment[i] = label

            # TODO
            norm = logsumexp(assignment_cost)
            posterior[i, :] = np.exp(assignment_cost - norm)

        edit_rate = edits / n_spots
        niter += 1

        # unique_assignment, cnts = np.unique(new_assignment, return_counts=True)

        # logger.info(f"Found ICM edit_rate={edit_rate:.6f} for iteration {niter}.")
        # logger.info(f"Found ICM inferred clone proportions: {cnts / n_spots}")

        if edit_rate < tol:
            break

    return niter, cost
