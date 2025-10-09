import numpy as np
import pytest
import logging
from cnaster.icm import (
    unpack_adjacency,
    build_wolff_cluster,
    wolff_update,
    wolff_sweep,
    calc_assignment_cost,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# NB 29.0838us -> 5.3694us with njit.
def test_build_wolff_cluster(benchmark):
    np.random.seed(42)

    # NB 0, 1 & 3 have the same assignment.
    new_assignment = np.array([0, 0, 1, 0, 1])
    n_spots = len(new_assignment)

    # NB 0, 1 & 3 are connected
    adjacency_spots = np.array([0, 0, 1, 1, 2, 3, 3, 4])
    adjacency_neighbors = np.array([1, 2, 0, 3, 4, 1, 4, 3])
    adjacency_weights = np.ones(adjacency_spots.shape[0])

    # NB deterministic behavior with always accepted.
    this_spot, p_add = 0, 1.0

    def run():
        build_wolff_cluster(
            new_assignment,
            adjacency_spots,
            adjacency_neighbors,
            adjacency_weights,
            this_spot,
            p_add,
        )

        return

    benchmark(run)

    cluster = build_wolff_cluster(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        p_add,
    )

    # With p_add=1.0, all connected spots with same assignment should be included
    # For this setup, spots 0, 1, and 3 are connected and have assignment 0.
    expected_cluster = np.array([0, 1, 3])

    assert set(cluster) == set(expected_cluster)
    assert cluster.dtype == np.int64
    assert cluster.shape[0] == len(expected_cluster)


# NB 34.64 us -> 5us.
def test_wolff_update(benchmark):
    np.random.seed(42)

    n_spots, n_clones, num_neighbors = 3_694, 3, 15

    single_llf = np.tile(np.arange(n_clones), n_spots).reshape(n_spots, n_clones)

    # NB must be symmetric.
    adjacency_list = [[] for _ in range(n_spots)]

    for i in range(n_spots):
        possible_neighbors = [j for j in range(n_spots) if j != i]
        chosen_neighbors = np.random.choice(
            possible_neighbors, size=num_neighbors, replace=False
        )

        for j in chosen_neighbors:
            # Add edge i -> j
            adjacency_list[i].append((j, 1.0 / num_neighbors))
            # Add edge j -> i (symmetric)
            adjacency_list[j].append((i, 1.0 / num_neighbors))

    # Remove duplicates (in case the same edge was added twice)
    for i in range(n_spots):
        unique_neighbors = list(set(adjacency_list[i]))
        adjacency_list[i] = unique_neighbors

    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adjacency_list)

    # new_assignment = np.random.randint(0, n_clones, size=n_spots)
    new_assignment = np.ones(n_spots, dtype=int)

    spatial_weight, p_add = 0.75, 0.1

    original_cost = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
    )

    posterior = np.zeros((n_spots, n_clones))

    new_cost, new_cluster_assignment, cluster = wolff_update(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        posterior,
        p_add=p_add,
        temp=np.inf,
        cost_zeropoint=original_cost,
    )
    """
    new_assignment[cluster] = new_cluster_assignment

    exp_cost = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
    )

    print(original_cost, new_cost, exp_cost)
    """

    def run():
        np.random.seed(42)
        """
        return wolff_update(
            single_llf,
            adj_spots,
            adj_neighbors,
            adj_weights,
            new_assignment,
            spatial_weight,
            posterior,
            p_add=p_add,
            temp=np.inf,
            cost_zeropoint=original_cost,
        )
        """
        return wolff_sweep(
            single_llf,
            adj_spots,
            adj_neighbors,
            adj_weights,
            new_assignment,
            spatial_weight,
            posterior,
            max_iter=10,
        )

    benchmark(run)
