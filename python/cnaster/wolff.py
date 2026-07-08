import numpy as np
from numba import njit
import scipy.sparse
from scipy.sparse.csgraph import connected_components
from cnaster.logger import get_logger
from cnaster.config import start_time
from cnaster.annotation import get_clone_label_annotation

logger = get_logger(__name__, start_time=start_time)


@njit(cache=True)
def _wolff_annealing_core(
    labels,
    single_llf,
    indptr,
    indices,
    weights,
    spatial_weight,
    anneal_temps,
    sweeps_per_temp,
):
    n_spots, n_clones = single_llf.shape

    in_cluster = np.zeros(n_spots, dtype=np.bool_)
    cluster_nodes = np.empty(n_spots, dtype=np.int32)
    queue = np.empty(n_spots, dtype=np.int32)
    cluster_llf = np.zeros(n_clones, dtype=np.float64)
    exp_logits = np.empty(n_clones, dtype=np.float64)

    for temp in anneal_temps:
        beta = 1.0 / temp

        base_J = 1.0 * spatial_weight
        p_add_base = 1.0 - np.exp(-beta * base_J)

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

            while q_head < q_tail:
                n = queue[q_head]
                q_head += 1

                start, end = indptr[n], indptr[n + 1]
                for i in range(start, end):
                    n_prime = indices[i]
                    if labels[n_prime] == mu and not in_cluster[n_prime]:
                        # J = spatial_weight * weights[i]
                        # p_add = 1.0 - np.exp(-beta * J)
                        if np.random.rand() <= p_add_base:
                            in_cluster[n_prime] = True
                            cluster_nodes[c_tail] = n_prime
                            c_tail += 1
                            queue[q_tail] = n_prime
                            q_tail += 1

            flips_this_temp += c_tail

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

            r = np.random.rand() * sum_exp
            cumsum = 0.0
            nu = n_clones - 1
            for k in range(n_clones):
                cumsum += exp_logits[k]
                if r <= cumsum:
                    nu = k
                    break

            print(
                "Temp:",
                round(temp, 4),
                "| Cluster size:",
                c_tail,
                "| Old label:",
                mu,
                "| New label:",
                nu,
            )

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
    num_temps=25,
    sweeps_per_temp=1,
    min_temp=1e-5,
    max_temp=None,
):
    labels = initial_assignment.copy()

    indptr = np.asarray(adj_indptr, dtype=np.int32)
    indices = np.asarray(adj_indices, dtype=np.int32)
    weights = np.asarray(adj_weights, dtype=np.float64)

    if max_temp is None:
        max_temp = spatial_weight * (weights.max() if weights.size > 0 else 1.0)

    anneal_temps = np.logspace(np.log10(min_temp), np.log10(max_temp), num=num_temps)[
        ::-1
    ]

    logger.info(f"Solving for temperature schedule:\n{anneal_temps}")

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


def initialize_clones_wolff(
    sample_ids,
    adjacency_mat,
    n_init=1,
    base_n_clones=10,
    spatial_weight=1.0,
    wolff_num_temps=500,
    wolff_sweeps_per_temp=1,
    min_temp=1e-5,
    max_temp=None,
    min_spots=None,
    relabel=False,
    random_state=None,
    config=None,
):
    if config is not None and config.annotation.clone_label is not None:
        assert (
            n_init == 1
        ), "Cannot generate multiple initializations when using a fixed clone label."

        clone_annotation, _ = get_clone_label_annotation(config)
        return clone_annotation

    if random_state is not None:
        np.random.seed(random_state)

    logger.info(
        f"Generating {n_init} wolff clone initializations,\n"
        f"base_n_clones={base_n_clones}, spatial_weight={spatial_weight}, min_temp={min_temp}, max_temp={max_temp}, relabel={relabel}."
    )

    logger.info(f"Expected cluster size={np.exp(spatial_weight)}")

    n_spots = len(sample_ids)
    all_initializations = []

    adj = adjacency_mat.tocsr()
    adj_indptr = adj.indptr
    adj_indices = adj.indices
    adj_weights = adj.data

    if relabel:
        row_indices = np.repeat(np.arange(n_spots), np.diff(adj_indptr))

    for i in range(n_init):
        logger.debug(f"Building Wolff initialization {i+1}/{n_init}")

        initial_assignment = np.random.randint(
            0, base_n_clones, size=n_spots, dtype=np.int32
        )
        single_llf = np.zeros((n_spots, base_n_clones), dtype=np.float64)

        smoothed_labels = wolff_sweep(
            single_llf,
            adj_indptr,
            adj_indices,
            adj_weights,
            initial_assignment,
            spatial_weight,
            num_temps=wolff_num_temps,
            sweeps_per_temp=wolff_sweeps_per_temp,
            min_temp=min_temp,
            max_temp=max_temp,
        )

        if relabel:
            valid_edges = (
                smoothed_labels[row_indices] == smoothed_labels[adj_indices]
            ) & (sample_ids[row_indices] == sample_ids[adj_indices])

            masked_weights = adj_weights[valid_edges]
            masked_rows = row_indices[valid_edges]
            masked_cols = adj_indices[valid_edges]

            masked_adj = scipy.sparse.csr_matrix(
                (masked_weights, (masked_rows, masked_cols)), shape=(n_spots, n_spots)
            )

            _, final_labels = connected_components(masked_adj, directed=False)
        else:
            final_labels = smoothed_labels

        initial_clone_index = []
        unique_labels = np.unique(final_labels)

        for c in unique_labels:
            idx = np.where(final_labels == c)[0]

            if min_spots is not None and len(idx) < min_spots:
                continue

            if len(idx) > 0:
                initial_clone_index.append(idx)

        all_initializations.append(initial_clone_index)

        logger.debug(
            f"Initialization {i+1} resulted in {len(initial_clone_index)} "
            f"distinct spatial clones."
        )

    logger.info(f"Successfully returned {n_init} initialized partitions.")
    return all_initializations
