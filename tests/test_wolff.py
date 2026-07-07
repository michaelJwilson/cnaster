import numpy as np
import pytest

# Adjust this import path to point to where you saved the optimized Wolff code
from cnaster.wolff import (
    build_csr_graph,
    wolff_update,
    wolff_sweep,
)

@pytest.fixture(scope="session")
def large_grid_graph_2d():
    """
    Creates a 50x50 2D square lattice graph (2,500 spots).
    A larger grid is used here to get meaningful benchmark timings.
    """
    L = 50
    n_spots = L * L
    spots, neighbors, weights = [], [], []
    
    for i in range(L):
        for j in range(L):
            node = i * L + j
            if i > 0:       
                spots.append(node); neighbors.append((i-1)*L + j); weights.append(1.0)
            if i < L - 1:   
                spots.append(node); neighbors.append((i+1)*L + j); weights.append(1.0)
            if j > 0:       
                spots.append(node); neighbors.append(i*L + j-1); weights.append(1.0)
            if j < L - 1:   
                spots.append(node); neighbors.append(i*L + j+1); weights.append(1.0)
                
    return (
        n_spots, 
        np.array(spots, dtype=np.int32), 
        np.array(neighbors, dtype=np.int32), 
        np.array(weights, dtype=np.float64)
    )

@pytest.fixture
def zero_field_setup(large_grid_graph_2d):
    n_spots, _, _, _ = large_grid_graph_2d
    n_clones = 4
    single_llf = np.zeros((n_spots, n_clones), dtype=np.float64)
    
    np.random.seed(42)
    initial_assignment = np.random.randint(0, n_clones, size=n_spots, dtype=np.int32)
    return single_llf, initial_assignment, 1.0

@pytest.fixture
def strong_field_setup(large_grid_graph_2d):
    n_spots, _, _, _ = large_grid_graph_2d
    n_clones = 2
    L = int(np.sqrt(n_spots))
    
    true_labels = np.zeros(n_spots, dtype=np.int32)
    for i in range(L):
        for j in range(L):
            if j >= L // 2:
                true_labels[i * L + j] = 1

    single_llf = np.zeros((n_spots, n_clones), dtype=np.float64)
    for n in range(n_spots):
        if true_labels[n] == 0:
            single_llf[n, 0] = 10.0
            single_llf[n, 1] = -10.0
        else:
            single_llf[n, 0] = -10.0
            single_llf[n, 1] = 10.0

    initial_assignment = 1 - true_labels
    return single_llf, initial_assignment, 1.5

def test_benchmark_csr_build(benchmark, large_grid_graph_2d):
    """
    Benchmarks the O(1) graph conversion utility. 
    This is critical to track, as it runs at the start of every sweep.
    """
    n_spots, adj_spots, adj_neighbors, adj_weights = large_grid_graph_2d
    
    build_csr_graph(n_spots, adj_spots, adj_neighbors, adj_weights)
    
    result = benchmark(build_csr_graph, n_spots, adj_spots, adj_neighbors, adj_weights)
    
    indptr, indices, weights = result
    assert len(indptr) == n_spots + 1
    assert len(indices) == len(adj_spots)


def test_benchmark_wolff_sweep_zero_field(benchmark, large_grid_graph_2d, zero_field_setup):
    """
    Benchmarks the entire annealed sweep in the purely spatial (zero-field) regime.
    Because clusters grow massively large here, this stresses the BFS queue memory limits.
    """
    n_spots, adj_spots, adj_neighbors, adj_weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight = zero_field_setup
    
    wolff_sweep(single_llf, adj_spots, adj_neighbors, adj_weights, initial_assignment, spatial_weight, sweeps_per_temp=1)

    benchmark(
        wolff_sweep, 
        single_llf, 
        adj_spots, 
        adj_neighbors, 
        adj_weights, 
        initial_assignment, 
        spatial_weight, 
        sweeps_per_temp=2
    )

def test_benchmark_wolff_sweep_strong_field(benchmark, large_grid_graph_2d, strong_field_setup):
    """
    Benchmarks the entire annealed sweep in the data-driven (strong-field) regime.
    Because external fields heavily restrict bond formation, clusters remain small, 
    stressing the array allocations/deallocations per step.
    """
    n_spots, adj_spots, adj_neighbors, adj_weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight = strong_field_setup
    
    wolff_sweep(single_llf, adj_spots, adj_neighbors, adj_weights, initial_assignment, spatial_weight, sweeps_per_temp=1)

    benchmark(
        wolff_sweep, 
        single_llf, 
        adj_spots, 
        adj_neighbors, 
        adj_weights, 
        initial_assignment, 
        spatial_weight, 
        sweeps_per_temp=2
    )