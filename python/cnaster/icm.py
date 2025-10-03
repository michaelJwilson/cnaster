import logging
import numpy as np
import csv
import time
from pathlib import Path
from scipy.special import logsumexp
from numba import njit
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


def get_clone_proportions(assignment):
    _, counts = np.unique(assignment, return_counts=True)
    return counts / len(assignment)


@dataclass
class HMRFPerfEntry:
    optimizer: str
    cost: float
    best_cost: float
    padd: float = np.nan
    iteration: int = 0
    clone_proportions: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_dict(self):
        d = asdict(self)
        d["cost"] = "{:+.6e}".format(self.cost)
        d["best_cost"] = "{:+.6e}".format(self.best_cost)
        d["is_best"] = self.cost == self.best_cost
        d["padd"] = "{:.2f}".format(self.padd) if not np.isnan(self.padd) else ""
        d["iteration"] = str(self.iteration)

        if isinstance(self.clone_proportions, np.ndarray):
            d["clone_proportions"] = ",".join(
                "{:.6f}".format(x) for x in self.clone_proportions
            )
        else:
            d["clone_proportions"] = str(self.clone_proportions)

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


def unpack_adjacency(adjacency_list):
    # TODO? hash map for O(1) lookup.
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


@njit
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
            if if neighbor not in cluster and (new_assignment[neighbor] == current_assignment):
                if np.random.rand() < p_add:
                    cluster.add(neighbor)
                    queue.append(neighbor)

    return np.array(list(cluster))


@njit
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


def wolff_update(
    single_llf,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    adjacency_list,  # TODO HACK
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
    p_add=0.5,
    cost_zeropoint=0.0,
    sample=False,
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
    logger.debug(
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

    # TODO HACK
    acceptance = np.exp(delta_cost)

    # NB we always accept the better state (max.)
    if delta_cost > 0:
        new_cluster_assignment = best_new_assignment
        new_cost = cost_zeropoint + delta_cost
        new_configuration = True

    # NB all proposed states are worse; pick one randomly;
    elif sample and (np.random.rand() < acceptance):
        new_cluster_assignment = best_new_assignment
        new_cost = cost_zeropoint + delta_cost
        new_configuration = True
    else:
        new_cluster_assignment, new_cost = current_assignment, cost_zeropoint
        new_configuration = False

    """
    logger.debug(
        f"Solved for better={int(delta_cost>0)} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {cost_zeropoint} -> {new_cost} @ acceptance={acceptance:.6e}"
    )
    """
    """
    # TODO define edits.
    for spin in cluster:
        new_assignment[spin] = new_cluster_assignment
    """
    new_assignment[cluster] = new_cluster_assignment

    return new_cost, new_configuration


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
    cost_zeropoint=0.0,
):
    """
    _, cost = icm_sweep(
        single_llf,
        adjacency_list,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        cost_zeropoint=cost_zeropoint,
    )

    HMRFPerfEntry(
        optimizer="icm",
        cost=cost,
        best_cost=cost,
        padd=np.nan,
        iteration=0,
        clone_proportions=get_clone_proportions(new_assignment),
    ).log()
    """
    unique_new_assignment = np.unique(new_assignment)

    # TODO HACK TEST
    new_assignment[:] = np.random.choice(
        unique_new_assignment, size=new_assignment.shape
    )

    logger.info(f"Solving for a Wolff sweep.")

    spots, neighbors, neighbor_weights = unpack_adjacency(adjacency_list)
    dp, best_cost = 0.05, -np.inf

    for p_add in np.arange(dp, 1.0 + dp, dp):
        for iteration in range(max_iter):
            cost, new_configuration = wolff_update(
                single_llf,
                spots,
                neighbors,
                neighbor_weights,
                adjacency_list,
                new_assignment,
                spatial_weight,
                posterior,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
                p_add=p_add,
                cost_zeropoint=cost,
            )

            if cost > best_cost:
                best_cost, best_assignment = cost, new_assignment.copy()
                logger.info(f"Found a new best assignment with cost={best_cost:.6e}")

            HMRFPerfEntry(
                optimizer="wolff",
                cost=cost,
                best_cost=best_cost,
                padd=p_add,
                iteration=iteration,
                clone_proportions=get_clone_proportions(new_assignment),
            ).log()
            """
            if new_configuration:
                _, cost = icm_sweep(
                    single_llf,
                    adjacency_list,
                    new_assignment,
                    spatial_weight,
                    posterior,
                    log_persample_weights=log_persample_weights,
                    sample_ids=sample_ids,
                    cost_zeropoint=cost,
                )

                if cost > best_cost:
                    best_cost, best_assignment = cost, new_assignment.copy()
                    logger.info(
                        f"Found a new best assignment with cost={best_cost:.6e}"
                    )

                HMRFPerfEntry(
                    optimizer="icm",
                    cost=cost,
                    best_cost=best_cost,
                    padd=np.nan,
                    iteration=-1,
                    clone_proportions=get_clone_proportions(new_assignment),
                ).log()
            """
    new_assignment[:] = best_assignment

    exit(0)

    return max_iter


# TODO
# @njit
def icm_sweep(
    single_llf,
    adjacency_list,
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
            for j, edge_weight in adjacency_list[i]:
                neighbor_assignment = new_assignment[j]
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

        _, cnts = np.unique(new_assignment, return_counts=True)

        logger.info(f"Found ICM edit_rate={edit_rate:.6f} for iteration {niter}.")
        logger.info(f"Found ICM inferred clone proportions: {cnts / n_spots}")

        if edit_rate < tol:
            break

    return niter, cost
