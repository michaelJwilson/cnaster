import numpy as np
from numba import njit
from cnaster.hmrf_utils import hmrf_perf_entry


@njit(cache=True)
def build_wolff_cluster(
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    this_spot,
    temp=1.0,
):
    """
    Construct a sub-cluster at a root spot with BFS from the cluster root.
    Addition to the sub_cluster occurs with prob.

    p_add = 1. - exp(-edge_weight / temp)

    if the neighbor has the same spin as the root.

    Returns the cluster and the log-probability of the (forward) move.
    """
    visited, cluster, queue = [this_spot], [this_spot], [this_spot]
    current_assignment = new_assignment[this_spot]

    lnprob_forward = 0.0

    while queue:
        current = queue.pop(0)

        mask = adjacency_spots == current
        neighbors = adjacency_neighbors[mask]
        weights = adjacency_weights[mask]

        for neighbor, edge_weight in zip(neighbors, weights):
            # NB we have considered this neighbor already - an effort
            #    to build smaller clusters with less cost and less
            #    book keeping.
            if neighbor in visited:
                continue
            else:
                visited.append(neighbor)

            # DEPRECATE a new neighbor by construction.
            if neighbor not in cluster and (
                new_assignment[neighbor] == current_assignment
            ):
                p_add = 1.0 - np.exp(-edge_weight / temp)

                if np.random.rand() <= p_add:
                    cluster.append(neighbor)
                    queue.append(neighbor)
                else:
                    lnprob_forward += np.log(1.0 - p_add)

    return np.array(sorted(list(cluster))), lnprob_forward


# DEPRECATE:  mis-guided effort for detailed balance by hand?  J energy cost
#             by construction, H energy cost by acceptance.
@njit(cache=True)
def get_cluster_lnprob_backward(
    new_assignment,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    this_spot,
    cluster,
    cluster_assignment,
    temp=1.0,
):
    visited, queue = [this_spot], [this_spot]
    lnprob_backward = 0.0

    while queue:
        current = queue.pop(0)

        mask = adjacency_spots == current
        neighbors = adjacency_neighbors[mask]
        weights = adjacency_weights[mask]

        for neighbor, edge_weight in zip(neighbors, weights):
            # NB we have considered this neighbor already - an effort
            #    to build smaller clusters with less cost.
            if neighbor in visited:
                continue
            else:
                visited.append(neighbor)

            # NB propagate wave front / frontier.
            if neighbor in cluster:
                queue.append(neighbor)
            elif new_assignment[neighbor] == cluster_assignment:
                p_add = 1.0 - np.exp(-edge_weight / temp)
                lnprob_backward += np.log(1.0 - p_add)

    return lnprob_backward


# TODO MOVE wolff
@njit(cache=True)
def wolff_update(
    single_llf,
    adjacency_spots,
    adjacency_neighbors,
    adjacency_weights,
    new_assignment,
    spatial_weight,
    log_persample_weights=None,
    sample_ids=None,
    cost_zeropoint=0.0,
    wolff_temp=None,
    anneal_temp=None,
):
    n_spots, n_clones = single_llf.shape

    # TODO smarter choice?
    this_spot = np.random.randint(n_spots)
    current_assignment = new_assignment[this_spot]

    # NB at high temp. returns a single spin by construction; at low temp. will return max.
    #    clique with shared spin.
    cluster, _ = build_wolff_cluster(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        wolff_temp,
    )
    """
    # NB equivalent to (locally) optimal ICM; return original cost and null op. cluster.
    if len(cluster) <= 1:
        # logger.info(f"Solved for a cluster of a single spin (equivalent to ICM).")
        return cost_zeropoint, current_assignment, None
    """
    # NB relative cost for assignment to each clone for posed cluster.
    node_cost, edge_cost = calc_cluster_assignment_cost(
        cluster,
        single_llf,
        sample_ids,
        log_persample_weights,
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        n_clones,
    )

    # NB assignment cost to each clone for this cluster.
    assignment_cost = node_cost + spatial_weight * edge_cost

    # logger.info(f"Found node and edge costs:\n{node_cost}\n{edge_cost}")

    current_cost = assignment_cost[current_assignment]

    # NB we look for the next 'best' ...
    assignment_cost[current_assignment] = -np.inf

    # NB Metropolis: if any assignment is lower, accept one randomly.
    #    otherwise, accept with exponential suppression, exp(-delta_cost).
    #
    # NB ignore "cost", we're solving for max.
    best_new_assignment = np.argmax(assignment_cost)
    delta_cost = assignment_cost[best_new_assignment] - current_cost

    # NB Metropolis step
    if delta_cost > 0:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment
        acceptance = 1.0

        return new_cost, new_cluster_assignment, cluster, acceptance
    """
    # NB sampled / exploration.
    lnprob_backward = get_cluster_lnprob_backward(
        new_assignment,
        adjacency_spots,
        adjacency_neighbors,
        adjacency_weights,
        this_spot,
        cluster,
        best_new_assignment,
        temp=temp,
    )
    """
    ln_acceptance = delta_cost / anneal_temp
    # ln_acceptance -= lnprob_forward
    # ln_acceptance += lnprob_forward

    acceptance = np.exp(ln_acceptance)
    accepted = np.random.rand() < acceptance
    """
    logger.info(
        f"Solved for a cluster of {len(cluster):4d}/{n_spots:4d} spins @ p_add={p_add:.3f} with current cost {current_cost:.4e},\
        next best cost={assignment_cost[best_new_assignment]:.4e} and dE={delta_cost:.4e}; accepted={accepted}."
    )
    """
    """                                                                                                                                                                                                               
    logger.info(                                                                                                                                                                                                       
        f"Solved for better={int(delta_cost>0)} cluster assignment {current_assignment} -> {new_cluster_assignment} with costs {cost_zeropoint} -> {new_cost}"                                                        
    )                                                                                                                                                                                                                 
    """

    # NB we always accept the better state (max.), a sampled state, or return the original.
    if accepted:
        new_cost = cost_zeropoint + delta_cost
        new_cluster_assignment = best_new_assignment

        return new_cost, new_cluster_assignment, cluster, acceptance
    else:
        return cost_zeropoint, current_assignment, cluster, acceptance


# TODO MOVE wolff
def wolff_sweep(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    log_persample_weights=None,
    sample_ids=None,
):
    original_assignment = new_assignment.copy()

    cost_zeropoint = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
    )

    hmrf_perf_entry(
        optimizer="zeropoint",
        cost=cost_zeropoint,
        best_cost=cost_zeropoint,
        temp=1.0,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    logger.info(f"Found an initial Potts cost={cost_zeropoint:.6e}.")

    # NB icm_sweep updates new_assignment in place; global max for T=np.inf (independent spins).
    _, new_cost = icm_sweep(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        cost_zeropoint=cost_zeropoint,
        temp=1.0,  # NB ICM is exact for independent spots, "high temperature".
    )

    hmrf_perf_entry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        temp=1.0,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    logger.info(
        f"Found a new best assignment with annealed ICM and new cost={new_cost:.6e}"
    )

    no_edge_cost = calc_assignment_cost(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        0.0,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
    )

    hmrf_perf_entry(
        optimizer="no_edge",
        cost=no_edge_cost,
        best_cost=new_cost,
        temp=np.inf,
        iteration=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    # NB unpacks adjaceny_list into arrays processble by numba.
    # TODO BUG? assumes ground state cost is higher than zero icm.
    best_assignment, best_cost = new_assignment.copy(), new_cost
    cost_zeropoint = best_cost

    high_temp = spatial_weight * adj_weights.max()

    # NB base 10 by default!
    # MAGIC HARDCODE
    anneal_temps = np.logspace(-7.0, 1.0 + np.log10(high_temp), num=5_000)[::-1]

    logger.info(
        f"Completing an annealed Wolff sweep with edge range=({adj_weights.min():.4f},{adj_weights.max():.4f}), high temperature {high_temp:.4e}, {len(anneal_temps)} decades:\n{anneal_temps}"
    )

    wolff_temp = 4.5
    max_iter = 0

    for anneal_iter, anneal_temp in enumerate(anneal_temps):
        logger.debug(f"Solving for anneal temperature {anneal_temp}.")

        new_assignment = best_assignment.copy()
        cost_zeropoint = best_cost

        for iteration in range(5):
            max_iter += 1

            new_cost, new_cluster_assignment, new_cluster, acceptance = wolff_update(
                single_llf,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
                cost_zeropoint=cost_zeropoint,
                wolff_temp=wolff_temp,  # NB effective cluster size for move.
                anneal_temp=anneal_temp,  # NB annealing of cost.
            )

            if new_cluster is not None:
                cost_zeropoint = new_cost

                for idx in new_cluster:
                    new_assignment[idx] = new_cluster_assignment

            if new_cost > best_cost:
                best_cost, best_assignment = new_cost, new_assignment.copy()

            if anneal_iter % 100 == 0:
                hmrf_perf_entry(
                    optimizer="wolff",
                    cost=new_cost,
                    best_cost=best_cost,
                    iteration=iteration,
                    temp=anneal_temp,
                    acceptance=acceptance,
                    ncluster=len(new_cluster) if new_cluster is not None else 0,
                    nedit=np.count_nonzero(new_assignment != original_assignment),
                    # clone_split=get_clone_split(new_assignment),
                ).log()

    # NB re-assign with the best found assignment.
    new_assignment[:] = best_assignment

    # NB temp=1. by default
    _, new_cost = icm_sweep(
        single_llf,
        adj_spots,
        adj_neighbors,
        adj_weights,
        new_assignment,
        spatial_weight,
        posterior,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
        cost_zeropoint=best_cost,
    )

    hmrf_perf_entry(
        optimizer="icm",
        cost=new_cost,
        best_cost=new_cost,
        iteration=0,
        temp=0.0,
        acceptance=1.0,
        ncluster=0,
        nedit=np.count_nonzero(new_assignment != original_assignment),
        # clone_split=get_clone_split(new_assignment),
    ).log()

    return max_iter, best_cost
