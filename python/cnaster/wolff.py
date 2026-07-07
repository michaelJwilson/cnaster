import numpy as np
import scipy.sparse

from numba import njit
import logging

logger = logging.getLogger(__name__)


@njit(cache=True)
def build_csr_graph(n_spots, adj_spots, adj_neighbors, adj_weights):
    degrees = np.zeros(n_spots, dtype=np.int32)
    for s in adj_spots:
        degrees[s] += 1
    indptr = np.zeros(n_spots + 1, dtype=np.int32)
    for i in range(n_spots):
        indptr[i + 1] = indptr[i] + degrees[i]
    indices = np.zeros_like(adj_neighbors)
    weights = np.zeros_like(adj_weights)
    current_pos = indptr[:-1].copy()
    for i in range(len(adj_spots)):
        s = adj_spots[i]
        pos = current_pos[s]
        indices[pos] = adj_neighbors[i]
        weights[pos] = adj_weights[i]
        current_pos[s] += 1
    return indptr, indices, weights


@njit(cache=True)
def wolff_update(
    labels, single_llf, indptr, indices, weights, spatial_weight, beta
):
    n_spots, n_clones = single_llf.shape
    rho = np.random.randint(n_spots)
    mu = labels[rho]

    in_cluster = np.zeros(n_spots, dtype=np.bool_)
    in_cluster[rho] = True

    cluster_nodes = np.empty(n_spots, dtype=np.int32)
    queue = np.empty(n_spots, dtype=np.int32)

    c_tail = 0
    cluster_nodes[c_tail] = rho
    c_tail += 1

    
    q_head, q_tail = 0, 0

    queue[q_tail] = rho
    q_tail += 1

    while q_head < q_tail:
        n = queue[q_head]
        q_head += 1

        start, end = indptr[n], indptr[n + 1]
        for i in range(start, end):
            n_prime = indices[i]
            if labels[n_prime] == mu and not in_cluster[n_prime]:
                J = spatial_weight * weights[i]
                p_add = 1.0 - np.exp(-beta * J)
                if np.random.rand() <= p_add:
                    in_cluster[n_prime] = True
                    cluster_nodes[c_tail] = n_prime
                    c_tail += 1
                    queue[q_tail] = n_prime
                    q_tail += 1

    nu = np.random.randint(n_clones)
    if nu == mu:
        return True, c_tail

    delta_llf = 0.0
    for i in range(c_tail):
        idx = cluster_nodes[i]
        delta_llf += single_llf[idx, nu] - single_llf[idx, mu]

    if delta_llf >= 0 or np.random.rand() <= np.exp(beta * delta_llf):
        for i in range(c_tail):
            labels[cluster_nodes[i]] = nu
        return True, c_tail

    return False, c_tail


def wolff_sweep(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    initial_assignment,
    spatial_weight,
    sweeps_per_temp=5,
):
    n_spots = single_llf.shape[0]
    labels = initial_assignment.copy()
    
    # indptr, indices, weights = build_csr_graph(
    #     n_spots, adj_spots, adj_neighbors, adj_weights
    # )

    csr = scipy.sparse.csr_matrix(
        (adj_weights, (adj_spots, adj_neighbors)), 
        shape=(n_spots, n_spots)
    )

    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    weights = np.asarray(csr.data, dtype=np.float64)

    # Accelerated schedule for testing purposes (fewer decades, fewer steps)
    # In production, use your np.logspace(-7.0, 1.0 + np.log10(high_temp), num=5_000)[::-1]
    high_temp = spatial_weight * (adj_weights.max() if adj_weights.size > 0 else 1.0)
    anneal_temps = np.logspace(-3.0, 1.0 + np.log10(high_temp), num=500)[::-1]

    for temp in anneal_temps:
        beta = 1.0 / temp
        for _ in range(sweeps_per_temp):
            wolff_update(
                labels, single_llf, indptr, indices, weights, spatial_weight, beta
            )

    return labels
