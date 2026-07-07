import numpy as np
import scipy.sparse
from numba import njit

@njit(cache=True)
def _wolff_annealing_core(
    labels, 
    single_llf, 
    indptr, 
    indices, 
    weights, 
    spatial_weight, 
    anneal_temps, 
    sweeps_per_temp
):
    """
    Fully JIT-compiled Wolff annealing engine.
    Uses the 'Informed Gibbs' Mega-Spin update and dynamically scales 
    cluster updates so exactly N spots are visited per sweep.
    """
    n_spots, n_clones = single_llf.shape

    # Pre-allocate all memory buffers ONCE for the entire annealing run.
    in_cluster = np.zeros(n_spots, dtype=np.bool_)
    cluster_nodes = np.empty(n_spots, dtype=np.int32)
    queue = np.empty(n_spots, dtype=np.int32)
    cluster_llf = np.zeros(n_clones, dtype=np.float64)
    exp_logits = np.empty(n_clones, dtype=np.float64)

    for temp in anneal_temps:
        beta = 1.0 / temp
        
        # A true "sweep" attempts to visit roughly N spots.
        target_flips = n_spots * sweeps_per_temp
        flips_this_temp = 0

        while flips_this_temp < target_flips:
            rho = np.random.randint(n_spots)
            mu = labels[rho]

            in_cluster[rho] = True
            c_tail = 0
            cluster_nodes[c_tail] = rho
            c_tail += 1

            q_head, q_tail = 0, 0
            queue[q_tail] = rho
            q_tail += 1

            # 1. Build the Cluster (Stochastic BFS)
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

            # Count these spots towards our sweep threshold
            flips_this_temp += c_tail

            # 2. Informed Gibbs Update
            # Reset and calculate aggregate log-likelihood for the mega-spin
            for k in range(n_clones):
                cluster_llf[k] = 0.0

            for i in range(c_tail):
                idx = cluster_nodes[i]
                for k in range(n_clones):
                    cluster_llf[k] += single_llf[idx, k]

            logits = beta * cluster_llf
            max_logit = np.max(logits)

            sum_exp = 0.0
            for k in range(n_clones):
                val = np.exp(logits[k] - max_logit)
                exp_logits[k] = val
                sum_exp += val

            # Sample new label safely
            r = np.random.rand() * sum_exp
            cumsum = 0.0
            nu = n_clones - 1
            for k in range(n_clones):
                cumsum += exp_logits[k]
                if r <= cumsum:
                    nu = k
                    break

            # 3. Apply assignment and reset mask
            if nu != mu:
                for i in range(c_tail):
                    labels[cluster_nodes[i]] = nu

            for i in range(c_tail):
                in_cluster[cluster_nodes[i]] = False

    return labels


def wolff_sweep(
    single_llf,
    adj_indptr,
    adj_indices,
    adj_weights,
    initial_assignment,
    spatial_weight,
    num_temps=500,
    sweeps_per_temp=2,
):
    n_spots = single_llf.shape[0]
    labels = initial_assignment.copy()

    # Force strict 32-bit ints for Numba compatibility
    indptr = np.asarray(adj_indptr, dtype=np.int32)
    indices = np.asarray(adj_indices, dtype=np.int32)
    weights = np.asarray(adj_weights, dtype=np.float64)

    high_temp = spatial_weight * (weights.max() if weights.size > 0 else 1.0)
    
    # Notice we drop to 1e-5 to guarantee a hard "quench" (greedy min) at the end
    anneal_temps = np.logspace(-5.0, 1.0 + np.log10(high_temp), num=num_temps)[::-1]

    # Hand off to the JIT engine
    _wolff_annealing_core(
        labels,
        single_llf,
        indptr,
        indices,
        weights,
        spatial_weight,
        anneal_temps,
        sweeps_per_temp,
    )

    return labels