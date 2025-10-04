import numpy as np
import pytest
from cnaster.icm import unpack_adjacency, build_wolff_cluster, wolff_update

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

    n_spots, n_clones = 5, 2
    single_llf = np.random.randn(n_spots, n_clones)

    # NB fully connected
    adjacency_list = [
        [(j, 1.0) for j in range(n_spots) if j != i] for i in range(n_spots)
    ]
    new_assignment = np.zeros(n_spots, dtype=int)
    spatial_weight = 0.5
    posterior = np.zeros((n_spots, n_clones))
    log_persample_weights = None
    sample_ids = np.zeros(n_spots, dtype=int)
    p_add = 0.5
    cost_zeropoint = 0.0

    spots, neighbors, weights = unpack_adjacency(adjacency_list)

    def run():
        return wolff_update(
            single_llf,
            spots,
            neighbors,
            weights,
            new_assignment,
            spatial_weight,
            posterior,
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
            p_add=p_add,
            cost_zeropoint=cost_zeropoint,
            temp=None,
        )

    new_cost, new_assignment, cluster = wolff_update(
        single_llf,
        spots,
        neighbors,
        weights,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        p_add=p_add,
        cost_zeropoint=cost_zeropoint,
        temp=None,
    )

    # assert np.isclose(new_cost, 0.44063562506550547)
    assert np.all(new_assignment == np.array([1, 1, 1, 1, 1]))
    assert np.all((new_assignment == 0) | (new_assignment == 1))

    benchmark(run)
