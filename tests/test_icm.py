import numpy as np
import pytest
from cnaster.icm import wolff_update, unpack_adjacency

# NB 34.64 us -> 5us.
def test_wolff_update(benchmark):
    np.random.seed(42)

    n_spots, n_clones = 5, 2
    single_llf = np.random.randn(n_spots, n_clones)

    adjacency_list = [
        [(j, 1.0) for j in range(n_spots) if j != i] for i in range(n_spots)
    ]  # fully connected
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
            adjacency_list,
            new_assignment,
            spatial_weight,
            posterior,
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
            p_add=p_add,
            cost_zeropoint=cost_zeropoint,
            sample=True,
        )

    new_cost, new_configuration = wolff_update(
        single_llf,
        spots,
        neighbors,
        weights,
        adjacency_list,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        p_add=p_add,
        cost_zeropoint=cost_zeropoint,
        sample=True,
    )

    # assert np.isclose(new_cost, 0.44063562506550547)
    assert np.all(new_assignment == np.array([1, 1, 1, 1, 1]))

    assert isinstance(new_configuration, bool)
    assert new_assignment.shape == (n_spots,)
    assert np.all((new_assignment == 0) | (new_assignment == 1))

    benchmark(run)
