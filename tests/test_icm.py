import numpy as np
import pytest
from cnaster.icm import wolff_update, unpack_adjacency

# NB 34.64 us -> 5us.
def test_wolff_update(benchmark):
    np.random.seed(42)

    n_spots, n_clones = 3694, 3

    # NB in the absence of spatial weight, single best clone in unary case.
    single_llf = np.eye(n_spots, n_clones)[:n_spots]

    MAX_NEIGHBORS = 10

    adjacency_list = []
    
    for i in range(n_spots):
        possible_neighbors = [j for j in range(n_spots) if j != i]
        num_neighbors = np.random.randint(1, MAX_NEIGHBORS + 1)
        chosen_neighbors = np.random.choice(possible_neighbors, size=num_neighbors, replace=False)
        adjacency_list.append([(j, 1.0) for j in chosen_neighbors])
    
    new_assignment = np.zeros(n_spots, dtype=int)
    posterior = np.zeros((n_spots, n_clones))
    log_persample_weights = None
    sample_ids = np.zeros(n_spots, dtype=int)

    spatial_weight, p_add, cost_zeropoint = 0.0, 0.1, 0.0
    min_acceptance = 0.0

    # TODO test unpack_adjacency.
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adjacency_list)
    
    def run():
        return wolff_update(
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
            cost_zeropoint=cost_zeropoint,
            min_acceptance=min_acceptance,
        )

    new_cost = wolff_update(
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
        cost_zeropoint=cost_zeropoint,
        min_acceptance=min_acceptance,
    )

    # assert np.isclose(new_cost, 0.44063562506550547)
    # assert np.all(new_assignment == np.array([1, 1, 1, 1, 1]))
    
    benchmark(run)
