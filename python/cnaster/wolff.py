import numpy as np
import scipy.sparse

from numba import njit
import logging

logger = logging.getLogger(__name__)


@njit(cache=True)
def wolff_update(labels, single_llf, indptr, indices, weights, spatial_weight, beta):
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

    q_head = 0
    q_tail = 0
    queue[q_tail] = rho
    q_tail += 1

    # 1. Build the cluster (Identical to before)
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

    # 2. INFORMED GIBBS UPDATE (Heat Bath for the Mega-Spin)
    # Calculate the aggregate log-likelihood for the entire cluster across all clones
    cluster_llf = np.zeros(n_clones, dtype=np.float64)
    for i in range(c_tail):
        idx = cluster_nodes[i]
        for k in range(n_clones):
            cluster_llf[k] += single_llf[idx, k]

    # Apply temperature scaling
    logits = beta * cluster_llf
    max_logit = np.max(logits)

    # Compute softmax probabilities safely (logsumexp trick to avoid numerical overflow)
    exp_logits = np.empty(n_clones, dtype=np.float64)
    sum_exp = 0.0
    for k in range(n_clones):
        val = np.exp(logits[k] - max_logit)
        exp_logits[k] = val
        sum_exp += val

    # 3. Sample a new label directly from the softmax distribution
    r = np.random.rand() * sum_exp
    cumsum = 0.0
    nu = n_clones - 1  # Fallback to last clone
    for k in range(n_clones):
        cumsum += exp_logits[k]
        if r <= cumsum:
            nu = k
            break

    # 4. Assign and strictly accept (Gibbs samples are always accepted)
    if nu != mu:
        for i in range(c_tail):
            labels[cluster_nodes[i]] = nu

    return True, c_tail


def wolff_sweep(
    single_llf,
    csr_indptr,
    csr_indices,
    csr_weights,
    initial_assignment,
    spatial_weight,
    num_temps=500,
    sweeps_per_temp=5,
):
    labels = initial_assignment.copy()
    high_temp = spatial_weight * (csr_weights.max() if csr_weights.size > 0 else 1.0)
    anneal_temps = np.logspace(-3.0, 1.0 + np.log10(high_temp), num=num_temps)[::-1]

    for temp in anneal_temps:
        beta = 1.0 / temp
        for _ in range(sweeps_per_temp):
            wolff_update(
                labels,
                single_llf,
                csr_indptr,
                csr_indices,
                csr_weights,
                spatial_weight,
                beta,
            )

    return labels
