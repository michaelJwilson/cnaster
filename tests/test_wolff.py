import logging

logging.disable(logging.INFO)

import numpy as np
import scipy.sparse
import pytest


from cnaster.icm import icm_sweep_deque
from cnaster.wolff import wolff_sweep

logging.getLogger("cnaster").setLevel(logging.WARNING)

def calculate_total_cost(assignment, single_llf, indptr, indices, weights, spatial_weight):
    """Calculates the objective function cost to fairly compare ICM and Wolff outputs."""
    cost = 0.0
    for i in range(len(assignment)):
        cost += single_llf[i, assignment[i]]
        start, end = indptr[i], indptr[i+1]
        for k in range(start, end):
            if assignment[i] == assignment[indices[k]]:
                cost += spatial_weight * weights[k]
    return cost

@pytest.fixture(scope="session")
def large_grid_graph_2d():
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
                
    csr = scipy.sparse.csr_matrix((weights, (spots, neighbors)), shape=(n_spots, n_spots))
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    data = np.asarray(csr.data, dtype=np.float64)
    
    return n_spots, indptr, indices, data

@pytest.fixture
def zero_field_setup(large_grid_graph_2d):
    n_spots, _, _, _ = large_grid_graph_2d
    n_clones = 4
    single_llf = np.zeros((n_spots, n_clones), dtype=np.float64)
    np.random.seed(42)
    initial_assignment = np.random.randint(0, n_clones, size=n_spots, dtype=np.int32)
    
    posterior = np.zeros_like(single_llf)
    return single_llf, initial_assignment, 1.0, posterior, None

@pytest.fixture
def strong_field_setup(large_grid_graph_2d):
    n_spots, _, _, _ = large_grid_graph_2d
    n_clones = 2
    L = int(np.sqrt(n_spots))
    
    true_labels = np.zeros(n_spots, dtype=np.int32)
    for i in range(L):
        for j in range(L):
            if j >= L // 2: true_labels[i * L + j] = 1

    single_llf = np.zeros((n_spots, n_clones), dtype=np.float64)
    for n in range(n_spots):
        if true_labels[n] == 0:
            single_llf[n, 0] = 10.0
            single_llf[n, 1] = -10.0
        else:
            single_llf[n, 0] = -10.0
            single_llf[n, 1] = 10.0

    initial_assignment = 1 - true_labels
    posterior = np.zeros_like(single_llf)
    return single_llf, initial_assignment, 1.5, posterior, true_labels

def test_benchmark_wolff_zero_field(benchmark, large_grid_graph_2d, zero_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, _, _ = zero_field_setup

    def run_wrapper():
        return wolff_sweep(single_llf, indptr, indices, weights, initial_assignment, spatial_weight, sweeps_per_temp=2)

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    majority_fraction = np.max(np.bincount(final_assignment)) / n_spots
    
    print(f"\n[Wolff Zero Field] Cost: {cost:,.2f} | Condensation: {majority_fraction*100:.1f}%")

def test_benchmark_icm_zero_field(benchmark, large_grid_graph_2d, zero_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, posterior, _ = zero_field_setup

    def run_wrapper():
        current_assignment = initial_assignment.copy()
        icm_sweep_deque(
            single_llf, indptr, indices, weights, current_assignment,
            spatial_weight, posterior, min_clone_spots=0
        )
        return current_assignment

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    majority_fraction = np.max(np.bincount(final_assignment)) / n_spots
    
    print(f"\n[ICM Zero Field] Cost: {cost:,.2f} | Condensation: {majority_fraction*100:.1f}%")

def test_benchmark_wolff_strong_field(benchmark, large_grid_graph_2d, strong_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, _, true_labels = strong_field_setup

    def run_wrapper():
        return wolff_sweep(single_llf, indptr, indices, weights, initial_assignment, spatial_weight, sweeps_per_temp=2)

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    accuracy = np.mean(final_assignment == true_labels) * 100
    
    print(f"\n[Wolff Strong Field] Cost: {cost:,.2f} | Accuracy: {accuracy:.2f}%")

def test_benchmark_icm_strong_field(benchmark, large_grid_graph_2d, strong_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, posterior, true_labels = strong_field_setup

    def run_wrapper():
        current_assignment = initial_assignment.copy()
        icm_sweep_deque(
            single_llf, indptr, indices, weights, current_assignment,
            spatial_weight, posterior, min_clone_spots=0
        )
        return current_assignment

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    accuracy = np.mean(final_assignment == true_labels) * 100
    
    print(f"\n[ICM Strong Field] Cost: {cost:,.2f} | Accuracy: {accuracy:.2f}%")