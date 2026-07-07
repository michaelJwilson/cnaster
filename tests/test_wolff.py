import numpy as np
import scipy.sparse
import pytest
# from numba import njit
# from collections import deque
# from scipy.special import logsumexp
# import logging

# logger = logging.getLogger(__name__)

from cnaster.icm import icm_sweep_deque
from cnaster.wolff import build_csr_graph, wolff_update, wolff_sweep


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
    
    # Create empty posterior for ICM
    posterior = np.zeros_like(single_llf)
    
    # We pass None for true_labels since it's zero field
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


# =============================================================================
# 4. BENCHMARK TESTS
# =============================================================================

# Global dictionary to hold custom metrics for the Pytest hook table
custom_metrics = {}

def test_benchmark_wolff_zero_field(benchmark, large_grid_graph_2d, zero_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, _, _ = zero_field_setup

    def run_wrapper():
        return wolff_sweep(single_llf, indptr, indices, weights, initial_assignment, spatial_weight, sweeps_per_temp=2)

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    majority_fraction = np.max(np.bincount(final_assignment)) / n_spots
    
    custom_metrics["Wolff Zero Field"] = {"Cost": cost, "Accuracy (%)": f"Condensation: {majority_fraction*100:.1f}%"}

def test_benchmark_icm_zero_field(benchmark, large_grid_graph_2d, zero_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, posterior, _ = zero_field_setup

    def run_wrapper():
        # Fresh copy needed every iteration so ICM doesn't start from an already solved state
        current_assignment = initial_assignment.copy()
        icm_sweep_deque(
            single_llf, indptr, indices, weights, current_assignment,
            spatial_weight, posterior, min_clone_spots=0
        )
        return current_assignment

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    majority_fraction = np.max(np.bincount(final_assignment)) / n_spots
    
    custom_metrics["ICM Zero Field"] = {"Cost": cost, "Accuracy (%)": f"Condensation: {majority_fraction*100:.1f}%"}


def test_benchmark_wolff_strong_field(benchmark, large_grid_graph_2d, strong_field_setup):
    n_spots, indptr, indices, weights = large_grid_graph_2d
    single_llf, initial_assignment, spatial_weight, _, true_labels = strong_field_setup

    def run_wrapper():
        return wolff_sweep(single_llf, indptr, indices, weights, initial_assignment, spatial_weight, sweeps_per_temp=2)

    final_assignment = benchmark.pedantic(run_wrapper, iterations=5, rounds=5)
    
    cost = calculate_total_cost(final_assignment, single_llf, indptr, indices, weights, spatial_weight)
    accuracy = np.mean(final_assignment == true_labels) * 100
    
    custom_metrics["Wolff Strong Field"] = {"Cost": cost, "Accuracy (%)": f"{accuracy:.2f}%"}

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
    
    custom_metrics["ICM Strong Field"] = {"Cost": cost, "Accuracy (%)": f"{accuracy:.2f}%"}


# =============================================================================
# 5. CUSTOM PYTEST TERMINAL HOOK (Renders the extra metrics table)
# =============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hooks into Pytest teardown to print a custom metrics table to the terminal."""
    if not custom_metrics:
        return
        
    terminalreporter.write("\n")
    terminalreporter.write_sep("=", "Custom Algorithm Metrics")
    terminalreporter.write_line(f"{'Benchmark Name':<30} | {'Final Cost (Energy)':<20} | {'Match / Condensation (%)'}")
    terminalreporter.write_line("-" * 80)
    
    for name, metrics in custom_metrics.items():
        cost_str = f"{metrics['Cost']:,.2f}"
        acc_str = metrics['Accuracy (%)']
        terminalreporter.write_line(f"{name:<30} | {cost_str:<20} | {acc_str}")
    terminalreporter.write_sep("=", "")