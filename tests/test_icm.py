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
    new_assignment = np.zeros((5,5), dtype=int)
    n_spots = len(new_assignment)

    for ii in range(1, 4):
        new_assignment[ii, 1:-1] = 1

    x,y = np.arange(5), np.arange(5)

    rows, cols = new_assignment.shape
    adj_spots_list, adj_neighbors_list = [],[]

    for r in range(rows):
        for c in range(cols):
            # NB C style row-major, matching numpy memory layout.
            i = c + r * cols
            
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr, cc = r + dr, c + dc
                
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = cc + rr * cols
                    
                    adj_spots_list.append(i)
                    adj_neighbors_list.append(j)

    # NB expected to be sorted; achieved by construction.
    adj_spots = np.array(adj_spots_list, dtype=int)
    adj_neighbors = np.array(adj_neighbors_list, dtype=int)

    # random weights per directed neighbor (seeded for reproducibility).
    rng = np.random.default_rng(0)
    adj_weights = rng.random(adj_spots.shape[0])

    new_assignment = new_assignment.flatten()

    # NB deterministic behavior with always accepted.
    this_spot = 12

    for temp, exp in zip([1.e-12, np.inf], [9, 1]):    
        cluster = build_wolff_cluster(
            new_assignment,
            adj_spots,
            adj_neighbors,
            adj_weights,
            this_spot,
            temp,
        )

        assert len(cluster) == exp
    
    def run():
        build_wolff_cluster(
            new_assignment,
            adj_spots,
            adj_neighbors,
            adj_weights,
            this_spot,
            temp,
        )

        return

    benchmark(run)


# NB 34.64 us -> 5us.
def test_wolff_update(benchmark):
    np.random.seed(42)

    n_spots, n_clones, num_neighbors = 3_694, 3, 15

    single_llf = np.tile(np.arange(n_clones), n_spots).reshape(n_spots, n_clones)

    # NB must be symmetric.
    adj_list = [[] for _ in range(n_spots)]

    for i in range(n_spots):
        possible_neighbors = [j for j in range(n_spots) if j != i]
        chosen_neighbors = np.random.choice(
            possible_neighbors, size=num_neighbors, replace=False
        )

        for j in chosen_neighbors:
            # Add edge i -> j
            adj_list[i].append((j, 1.0 / num_neighbors))
            # Add edge j -> i (symmetric)
            adj_list[j].append((i, 1.0 / num_neighbors))

    # Remove duplicates (in case the same edge was added twice)
    for i in range(n_spots):
        unique_neighbors = list(set(adj_list[i]))
        adj_list[i] = unique_neighbors

    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

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
        p_add=p_add,
        temp=np.inf,
        cost_zeropoint=original_cost,
    )

    new_assignment[cluster] = new_cluster_assignment

    exp_cost = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
    )

    assert np.isclose(new_cost, exp_cost, rtol=1e-9, atol=1e-6)

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

    # benchmark(run)
