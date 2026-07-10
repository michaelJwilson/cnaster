import copy
import time

import numpy as np

# import pandas as pd
import scipy.special
from numba import njit, prange

# from pathlib import Path
from cnaster.icm import (
    # icm_sweep,
    icm_sweep_deque,
    # icm_sweep_pqueue,
    unpack_adjacency,
    merge_assignment,
)

# from cnaster.wolff import wolff_sweep
from cnaster.hmm_initialize import gmm_init, cna_mixture_init
from cnaster.hmm import pipeline_baum_welch

# from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmm_phased import hmm_phased
from cnaster.hmrf_utils import cast_csr, clone_stack_obs

# from cnaster.utils import count_calls, get_output_dir, write_fig
# from cnaster.hmm_initialize import plot_cna_mixture, cna_mixture_init
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.config import get_global_config

# from cnaster.plotting import plot_clones_spatial
# from cnaster.plot_genomic import plot_clones_genomic
# from cnaster.deprecated.hmrf import (
#     aggr_hmrfmix_reassignment_concatenate as dep_aggr_hmrfmix_reassignment_concatenate,
# )
from sklearn.metrics import adjusted_rand_score
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


@njit
def logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def validate_clone_ids(assignments):
    unique_ids = np.unique(assignments)

    # NB check that clone ids are contiguous, i.e. 0,1,...,n_clones-1
    expected = np.arange(len(unique_ids))

    if not np.array_equal(unique_ids, expected):
        logger.error(f"Found invalid clone ids (e.g. not contiguous): {unique_ids}.")
        raise RuntimeError()

    return True

# TODO
@njit(cache=True)
def pool_hmrf_data(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    smooth_indices,
    smooth_indptr,
    single_tumor_prop=None,
    is_tumor_mixed=False,
    res_new_log_mu=None,
    pred=None,
    n_states=None,
    lambd=None,
):
    # NB no logger comments in jit compiled.
    n_obs, n_comp, N = single_X.shape

    pooled_X = np.zeros((n_obs, n_comp, N), dtype=single_X.dtype)
    pooled_base_nb_mean = np.zeros((n_obs, N), dtype=single_base_nb_mean.dtype)
    pooled_total_bb_RD = np.zeros((n_obs, N), dtype=single_total_bb_RD.dtype)
    mean_tumor_prop, weighted_tp = None, None
    """
    # TODO HACK BUG  
    if is_tumor_mixed:
        n_clones = int(len(pred) / n_obs)
        
        assert res_new_log_mu.shape[-1] == n_clones

        mean_tumor_prop = np.zeros(N, dtype=np.float64)
        
        weighted_mu = np.zeros((n_obs, n_clones), dtype=np.float64)
        weighted_tp = np.zeros((n_obs, n_clones, N), dtype=np.float64)
        
        for c in range(n_clones):
            norm = 0.0

            for obs_idx in range(n_obs):
                # NB modulo phasing.
                state_idx = pred[c * n_obs + obs_idx] % n_states
                mu = np.exp(res_new_log_mu[state_idx, c])
                norm += mu * lambd[obs_idx]
                
            for obs_idx in range(n_obs):
                state_idx = pred[c * n_obs + obs_idx] % n_states
                mu = np.exp(res_new_log_mu[state_idx, c])
            
                weighted_mu[obs_idx, c] = mu / norm
    """
    for i in range(N):
        start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

        # NB vaild neighbors have finite tumor proportion if is_tumor_mixed.
        valid_neighbors = []

        for k in range(start_idx, end_idx):
            col = smooth_indices[k]

            if is_tumor_mixed and single_tumor_prop is not None:
                if not np.isnan(single_tumor_prop[col]):
                    valid_neighbors.append(col)
            else:
                valid_neighbors.append(col)

        valid_count = len(valid_neighbors)

        # TODO CHECK prior behavior?
        #
        # NB assigned zero to pooled_X, pooled_base_nb_mean, pooled_total_bb_RD
        #    if no valid neighbors.
        if valid_count == 0:
            continue

        # NB for all segments, and spots, we pool (sum) the X, base_nb_mean, and total_bb_RD
        #    of all valid neighbors.
        #
        #    valid as updated the counts and the baseline will be accounted for in the likelihood (TBC).
        for obs_idx in range(n_obs):
            for neighbor_idx in valid_neighbors:
                pooled_X[obs_idx, 0, i] += single_X[obs_idx, 0, neighbor_idx]
                pooled_X[obs_idx, 1, i] += single_X[obs_idx, 1, neighbor_idx]

                pooled_base_nb_mean[obs_idx, i] += single_base_nb_mean[
                    obs_idx, neighbor_idx
                ]

                pooled_total_bb_RD[obs_idx, i] += single_total_bb_RD[
                    obs_idx, neighbor_idx
                ]
        """
        # TODO HACK BUG we do not currently support tumor proportion.
        if is_tumor_mixed:
            # NB calculate the 
            tumor_prop_sum = 0.0

            for neighbor_idx in valid_neighbors:
                tumor_prop_sum += single_tumor_prop[neighbor_idx]

            # NB input to compute_emission_probability_nb_betabinom_mix
            mean_tumor_prop[i] = tumor_prop_sum / valid_count

            if np.sum(pooled_base_nb_mean[:, i]) > 0:
                for c in range(n_clones):
                    for obs_idx in range(n_obs):
                        weighted_tp[obs_idx, c, i] = (
                            mean_tumor_prop[i] * weighted_mu[obs_idx, c]
                        ) / (
                            mean_tumor_prop[i] * weighted_mu[obs_idx, c]
                            + 1.0
                            - mean_tumor_prop[i]
                        )
            else:
                for obs_idx in range(n_obs):
                    weighted_tp[obs_idx, c, i] = mean_tumor_prop[i]
        """
    return (
        pooled_X,
        pooled_base_nb_mean,
        pooled_total_bb_RD,
        mean_tumor_prop,
        weighted_tp,
    )

# TODO 
@njit(parallel=True, cache=True)
def compute_single_llf(
    N,
    nz_nb_base,
    nz_bb_total,
    single_tumor_prop,
    is_tumor_mixed,
    tmp_log_emission_rdr,
    tmp_log_emission_baf,
    pred,
    n_obs,
    n_clones,
    smooth_indices=None,
    smooth_indptr=None,
    non_zero_weight=True,  # NB if False, rdr out-weighs the baf signal, which has many zero read_depth segments.
):
    # NB compute the log likelihood for each spot, for all clones.
    single_llf = np.zeros((N, n_clones))
    ratio_nonzeros = np.ones(N, dtype=np.float64)

    if non_zero_weight and smooth_indices is not None and smooth_indptr is not None:
        for i in prange(N):
            start_idx, end_idx = smooth_indptr[i], smooth_indptr[i + 1]

            # NB calculate the nb and bb baseline for this spot.
            sum_nb_base = 0.0
            sum_bb_total = 0.0

            # NB loop over pooled neighbors of spot i,
            #    skipping those with nan tumor proportion.
            for k in range(start_idx, end_idx):
                neighbor = smooth_indices[k]

                if is_tumor_mixed:
                    if np.isnan(single_tumor_prop[neighbor]):
                        continue

                # NB nz_nb_base, nz_bb_total contain the number of non-zero genomic segments;
                #    pools this across spots.
                sum_nb_base += nz_nb_base[neighbor]
                sum_bb_total += nz_bb_total[neighbor]

            # NB both normal and baf signals available.
            if sum_nb_base > 0 and sum_bb_total > 0:
                ratio_nonzeros[i] = sum_bb_total / sum_nb_base

    # NB Numba evaluates .ndim at compile-time. This creates a zero-cost branch.
    is_1d_pred = pred.ndim == 1

    for i in prange(N):
        for c in range(n_clones):
            term_rdr = 0.0
            term_baf = 0.0

            for o in range(n_obs):
                # NB Route the indexing dynamically based on array shape
                if is_1d_pred:
                    copy_state = pred[c * n_obs + o]
                else:
                    copy_state = pred[o, c]

                term_rdr += tmp_log_emission_rdr[copy_state, o, i]
                term_baf += tmp_log_emission_baf[copy_state, o, i]

            single_llf[i, c] = ratio_nonzeros[i] * term_rdr + term_baf

    return single_llf


# NB aggregate by smooth mat. with tumor/normal mix, spot reassignment, concatenated by clone?
def pipeline_clone_assignment(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    pred,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    spatial_weight,
    smooth_mat=None,
    log_persample_weights=None,
    single_tumor_prop=None,
    hmmclass=None,
    return_posterior=False,
    merge=False,
):
    # NB n_obs is the number of genomic segments, N is the number of spots.
    n_obs, _, N = single_X.shape
    n_states = res["new_p_binom"].shape[0]

    # NB pred is the argmax posterior by genome, potentially __concatenated__ across clones.
    if pred.ndim == 1:
        n_clones = len(pred) // n_obs
    else:
        n_clones = pred.shape[1]

    start_time = time.time()

    # NB clone assignment for all spots.
    new_assignment = copy.copy(prev_assignment)

    # NB utilize tumor mixture model?
    is_tumor_mixed = single_tumor_prop is not None

    # NB compute lambda, i.e. normalized baseline expression, for mixture model
    lambd = (
        np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
        if is_tumor_mixed
        else None  # TODO BUG?
    )

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={single_X.shape}, n_states={n_states} and {n_clones} clones with {hmmclass.__name__}, is_tumor_mixed={is_tumor_mixed} and merge={merge}."
    )

    logger.info("Pooling hmrf data by smooth mat. (reduces necessary computation).")

    if smooth_mat is not None:
        # NB   pool (sum) data according to smooth (adjacency) matrix, for X, nb_baseline, bb read depth and mean tumor proportion:
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, weighted_tp = (
            pool_hmrf_data(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                smooth_mat.indices,
                smooth_mat.indptr,
                single_tumor_prop,
                is_tumor_mixed,
                res["new_log_mu"] if is_tumor_mixed else None,
                pred if is_tumor_mixed else None,
                n_states if is_tumor_mixed else None,
                lambd,
            )
        )
    else:
        # TODO copies necessary?
        pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, weighted_tp = (
            single_X.copy(),
            single_base_nb_mean.copy(),
            single_total_bb_RD.copy(),
            None,
            None,
        )

    # NB emission shape: (n_states, n_obs, n_spots)
    if is_tumor_mixed:
        (
            tmp_log_emission_rdr,
            tmp_log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
            pooled_X,
            pooled_base_nb_mean,
            res["new_log_mu"],
            res["new_alphas"],
            pooled_total_bb_RD,
            res["new_p_binom"],
            res["new_taus"],
            np.ones((n_obs, 1))
            * np.mean(single_tumor_prop[idx]),  # TODO BUG  idx is not defined (!)
            weighted_tp.reshape(-1, 1),  # NB cast (n_obs,) to (n_obs, 1).
        )
    else:
        (
            tmp_log_emission_rdr,
            tmp_log_emission_baf,
        ) = hmmclass.compute_emission_probability_nb_betabinom(
            pooled_X,
            pooled_base_nb_mean,
            res["new_log_mu"],
            res["new_alphas"],
            pooled_total_bb_RD,
            res["new_p_binom"],
            res["new_taus"],
        )

    """
    # NB log likelihood of each spot given that its label is each clone, i.e. unary Potts term.
    single_llf = np.zeros((N, n_clones))

    # TODO numba
    for i in range(N):
        # NB spot i pooled with 'neighbor' spots in idx.
        idx = smooth_mat[i, :].nonzero()[1]

        if use_mixture:
            # NB filter out NaN tumor proportions
            idx = idx[~np.isnan(single_tumor_prop[idx])]

        # TODO pooled_X treated as a bool
        # NB pooled neighbors have at least one normal umi, and at least one snp-covering umi (all segments)
        if (
            np.sum(single_base_nb_mean[:, idx] > 0) > 0
            and np.sum(single_total_bb_RD[:, idx] > 0) > 0
        ):
            # NB aggregating over all segments in these neighbors, ratio of normal umis to snp-covering umis
            #    for pooled neighbors of spot i.
            ratio_nonzeros = (
                1.0
                * np.sum(single_total_bb_RD[:, idx] > 0)
                / np.sum(single_base_nb_mean[:, idx] > 0)
            )

            for c in range(n_clones):
                # NB MAP copy state for this clone (concatenated).
                this_pred = pred[(c * n_obs) : ((c + 1) * n_obs)]

                # NB log likelihood for this spot, given copy number
                #    profile of this clone; assumes IID along the genome.
                single_llf[i, c] = ratio_nonzeros * np.sum(
                    tmp_log_emission_rdr[this_pred, np.arange(n_obs), i]
                ) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), i])
        else:
            for c in range(n_clones):
                this_pred = pred[(c * n_obs) : ((c + 1) * n_obs)]

                single_llf[i, c] = np.sum(
                    tmp_log_emission_rdr[this_pred, np.arange(n_obs), i]
                ) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), i])
    """
    _tumor_prop = single_tumor_prop if single_tumor_prop is not None else np.empty(0)

    # NB number of non-zero genomic segments for nb_baseline and bb_read depth, for all spots.
    nz_nb_base = (single_base_nb_mean > 0).sum(axis=0)
    nz_bb_total = (single_total_bb_RD > 0).sum(axis=0)

    # NB computes the log likelihood for each spot, for all clones, given the "pooling" strategy,
    #    no longer IID and erroneously weights rdr and baf according to number of non-zero segments.
    single_llf = compute_single_llf(
        N,
        nz_nb_base,
        nz_bb_total,
        _tumor_prop,
        is_tumor_mixed,
        tmp_log_emission_rdr,
        tmp_log_emission_baf,
        pred,
        n_obs,
        n_clones,
        smooth_indices=smooth_mat.indices if smooth_mat is not None else None,
        smooth_indptr=smooth_mat.indptr if smooth_mat is not None else None,
    )

    # assert np.allclose(single_llf, new_single_llf), "BUG: single_llf mismatch"

    adj_list = cast_csr(adjacency_mat)
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

    # NB Posterior probabilities if return_posterior=True.
    posterior = np.zeros((N, n_clones))

    if get_global_config().hmrf.fixed_assignment:
        logger.warning(f"Assuming a fixed clone assignment")
    else:
        logger.info(f"Solving for updated clone assignment with icm_sweep_dequeue.")

        # NB updates new_assignment and posterior in place given log emission likelihood.
        """
        niter, new_cost = icm_sweep_deque(
            single_llf,
            adj_spots,
            adj_neighbors,
            adj_weights,
            new_assignment,
            spatial_weight,
            posterior,
            # tol=0.1,  # MAGIC TODO
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
        )
        """
        niter, new_cost = icm_sweep_deque(
            single_llf=single_llf,
            adj_indptr=adjacency_mat.indptr,
            adj_indices=adjacency_mat.indices,
            adj_weights=adjacency_mat.data,
            new_assignment=new_assignment,
            spatial_weight=spatial_weight,
            posterior=posterior,
            onehot_allowed_clones=None,
            # tol=0.1,  # MAGIC TODO
            log_persample_weights=log_persample_weights,
            sample_ids=sample_ids,
        )
        
        logger.info(f"Ready for potential merging of clones?  {merge}.")

        while merge:
            # NB merge_assignment returns the original cost, the best new cost after merging this clone pair, and the clone pair.
            new_cost, best_merge_cost, best_merge_pair = merge_assignment(
                single_llf,
                adj_spots,
                adj_neighbors,
                adj_weights,
                new_assignment,
                spatial_weight,
                log_persample_weights=log_persample_weights,
                sample_ids=sample_ids,
            )

            # NB only merge the clone pair if the cost is improved.
            if best_merge_cost > new_cost:
                u, v = best_merge_pair
                num_merged_spots = 0

                # TODO ensure clone "v" has a larger index than clone "u" to minimize downstream reindexing issues.
                # NB assigns clone "u" to clone "v"
                for i in range(len(new_assignment)):
                    if new_assignment[i] == u:
                        new_assignment[i] = v
                        num_merged_spots += 1

                logger.info(
                    f"Merged {num_merged_spots} spots from clone {u} into clone {v} with new cost={best_merge_cost} given original cost={new_cost:.6e}."
                )

                # TODO DEPRECATE?
                new_cost = best_merge_cost
            else:
                logger.info(
                    f"No more beneficial merges available (best merge cost={best_merge_cost} given original cost={new_cost:.6e})."
                )
                break

        # NB counts per clone in the final (potentially merged) assignment.
        _, cnts = np.unique(new_assignment, return_counts=True)

        logger.info(
            f"Found new clone assignment with new cost {new_cost:.6e} in {niter} iterations ({time.time() - start_time:.2f}s with clone breakdown=\n{[f'{xx:.3f}' for xx in cnts / cnts.sum()]})."
        )

    logger.info(f"Computing total ln likelihood.")

    # NB single_llf was the log likelihood of each spot given that its label is each clone, i.e. unary Potts term;
    #    sum this assuming iid given new assignment.
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])

    # NB add the pairwise cost for this assignment, according to the (weighted) number of neighbors with the same assignment.
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )

    """
    # TODO HACK?  pred of HMM requires an clone ordering definition?
    # 
    # NB reindex new_assignment to contiguous clone ids.
    unique_ids = np.unique(new_assignment)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    new_assignment = np.array([id_map[x] for x in new_assignment])
    """
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf


# @count_calls
def run_core_inference(
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    initial_clone_index,
    n_states,
    log_sitewise_transmat,
    prefix="clones",
    coords=None,
    smooth_mat=None,
    adjacency_mat=None,
    sample_ids=None,
    sample_list=None,
    max_iter_outer=5,
    # nodepotential="max",
    hmmclass=hmm_phased,  # hmm_sitewise
    hmm_initializer=gmm_init,  # {cna_mixture_init, gmm_init}
    params="stmp",
    t=1 - 1e-6,
    random_state=0,
    init_log_mu=None,
    init_p_binom=None,
    init_alphas=None,
    init_taus=None,
    fix_NB_dispersion=False,
    shared_NB_dispersion=True,
    fix_BB_dispersion=False,
    shared_BB_dispersion=True,
    is_diag=True,
    max_iter=100,
    tol=1e-4,
    # unit_xsquared=9,
    # unit_ysquared=3,
    spatial_weight=1.0 / 6.0,
    tumorprop_threshold=0.5,
    plot_progress=False,
    propagate_hmm_param_errors=False,
    deconcatenate_clones=False,
):
    # NB num. of genomic bins, num. pseudobulk (clones, spots, ...)
    n_obs, _, _ = single_X.shape

    # NB num. of clones in initial assignment.
    n_clones = len(initial_clone_index)

    # NB map sample_ids to integer enum, i.e. per slice.
    unique_sample_ids = np.unique(sample_ids)
    n_samples = len(unique_sample_ids)

    logger.info(
        f"Running hmrfmix_concatenate_pipeline for {n_clones} clones and {n_samples} samples/slices."
    )

    tmp_map_index = {unique_sample_ids[i]: i for i in range(len(unique_sample_ids))}
    sample_ids = np.array([tmp_map_index[x] for x in sample_ids])

    norm = np.sum(single_base_nb_mean)

    # DEPRECATE
    assert np.isscalar(norm)

    # NB baseline expression by summing over all clones; should be zero for BAF only.
    if norm == 0.0:
        logger.warning(
            f"Found nb_mean=0 across all spots,segments; corresponds to BAF only run."
        )

    # NB normalized baseline expression;
    with np.errstate(divide="ignore", invalid="ignore"):
        lambd = np.sum(single_base_nb_mean, axis=1) / norm

    # NB aggregation to pseudobulk based on current clone assignment of spots.
    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index,
        single_tumor_prop,
        threshold=tumorprop_threshold,
    )

    # validation_summary(lengths, X, base_nb_mean, total_bb_RD, tumor_prop)

    # NB transform (n_obs, 2, n_clones) to (n_obs * n_clones, 2, 1) for HMM processing.
    #    i.e. stack bins per clone lengthwise, useful for fitting shared copy state.
    (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_lengths,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
    ) = clone_stack_obs(
        X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
    )

    merge = False

    if (init_log_mu is None) or (init_p_binom is None):
        # TODO HACK
        transmat = np.ones((n_states, n_states)) * (1.0 - t) / (n_states - 1)
        np.fill_diagonal(transmat, t)
        log_transmat = np.log(transmat)

        new_init_log_mu, new_init_p_binom, _, _ = hmm_initializer(
            n_states,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            params,
            clone_stack_lengths,
            log_transmat,
            clone_stack_sitewise_transmat,
            random_state=random_state,
            in_log_space=False,
            only_minor=False,  # NB with no phasing, we need states > 0.5;
        )

        new_init_alphas = init_alphas
        new_init_taus = init_taus

        """
        new_init_log_mu, new_init_alphas, new_init_p_binom, new_init_taus = (
            cna_mixture_init(
                n_states,
                clone_stack_X,
                clone_stack_base_nb_mean,
                clone_stack_total_bb_RD,
                width=10,
            )
        )
        """

        if init_log_mu is None:
            init_log_mu = new_init_log_mu
            init_alphas = new_init_alphas

        if init_p_binom is None:
            init_p_binom = new_init_p_binom
            init_taus = new_init_taus

        logger.info(
            f"Solved for hmm initialized parameters:\n{init_log_mu}\n{init_p_binom}"
        )

        # logger.info(
        #     f"Plotting initial copy state mixture for instance {hmrfmix_concatenate_pipeline.call_count-1} with X.shape={X.shape}."
        # )

        n_states = init_p_binom.shape[0]

    last_log_mu = init_log_mu if "m" in params else None
    last_p_binom = init_p_binom if "p" in params else None
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)

    for c, idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # NB inertia to spot clone change: log(1 / n_clones) per spot, i.e. uniform prior over clones.
    inertia = bool(get_global_config().hmrf.inertia)
    log_persample_weights = (
        np.ones((n_clones, n_samples)) * (-np.log(n_clones)) if inertia else None
    )

    logger.info(f"Assuming hmrf inertia={inertia} and {hmmclass.__name__} instance.")

    # TODO HACK this scratches res input.
    # NB res required for remain_kwargs construction;
    res = {}
    r = 0

    # NB convoluted loop logic to achieve merge on last iteration.
    while r <= max_iter_outer:
        logger.info(
            f"----****  Solving iteration {r}/{max_iter_outer} of copy number state fitting & clone assignment (HMM + HMRF) ****----"
        )

        # NB [num_segments, num_segments ..., num_segments] of length num_clones.
        sample_length = X.shape[0] * np.ones(X.shape[2], dtype=int)
        remain_kwargs = {"sample_length": sample_length, "lambd": lambd}

        """
        # TODO HACK BUG?
        # NB utilize last state posterior. 
        if "log_gamma" in res:
            remain_kwargs["log_gamma"] = res["log_gamma"]
        """
        res = pipeline_baum_welch(
            None,
            clone_stack_X,
            clone_stack_lengths,
            n_states,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            clone_stack_sitewise_transmat,
            stack_tumor_prop,
            hmmclass=hmmclass,
            params=params,
            t=t,
            random_state=random_state,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            is_diag=is_diag,
            init_log_mu=last_log_mu,
            init_p_binom=last_p_binom,
            init_alphas=last_alphas,
            init_taus=last_taus,
            max_iter=max_iter,
            tol=tol,
            **remain_kwargs,
        )

        # NB MAP copy state, irrespective of phasing. contrast to "pred_cnv".
        pred = np.argmax(res["log_gamma"], axis=0)

        # NB TODO 'max' clone assignment.
        new_assignment, _, total_llf = pipeline_clone_assignment(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            adjacency_mat,
            last_assignment,
            sample_ids,
            smooth_mat=smooth_mat,
            spatial_weight=spatial_weight,
            log_persample_weights=log_persample_weights,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
            merge=merge,
        )
        """
        # NB new assignment did not populate an input clone.
        if len(np.unique(new_assignment)) < X.shape[2]:
            # DEPRECATE
            # res["assignment_before_reindex"] = new_assignment

            # NB imposes new order, if not previously sorted, rather than skip only.
            remaining_clones = np.sort(np.unique(new_assignment))

            # NB map clone id -> new (0,...,N-1) enumeration.
            re_indexing = {c: i for i, c in enumerate(remaining_clones)}

            logger.warning(
                f"Detected clone loss on iteration {r}:  re-indexing clones with {re_indexing}"
            )

            # NB re-index new_assignment to be consecutive given a missing clone.
            # TODO faster way?
            new_assignment = np.array([re_indexing[x] for x in new_assignment])

            concat_idx = np.concatenate(
                [np.arange(c * n_obs, c * n_obs + n_obs) for c in remaining_clones]
            )

            # NB log_gamma and pred_cnv by new clone order (concatenated).
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]
        """
        remaining_clones, new_assignment_reindexed = np.unique(
            new_assignment, return_inverse=True
        )

        if len(remaining_clones) < X.shape[2]:
            logger.warning(f"Detected clone loss on iteration {r}: re-indexing clones.")

            new_assignment = new_assignment_reindexed
            concat_idx = (remaining_clones[:, None] * n_obs + np.arange(n_obs)).ravel()

            # NB log_gamma and pred_cnv by new clone order (concatenated).
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]

        res["prev_assignment"] = last_assignment
        res["new_assignment"] = new_assignment
        res["total_llf"] = total_llf

        clone_index = [
            np.where(res["new_assignment"] == c)[0]
            for c in np.unique(res["new_assignment"])
        ]

        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )

        # TODO clone stack can be an arg. to merge_pseudobulk_by_index_mix
        (
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            clone_stack_lengths,
            clone_stack_sitewise_transmat,
            stack_tumor_prop,
        ) = clone_stack_obs(
            X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
        )

        state_counts = np.bincount(pred, minlength=n_states)
        state_usage = state_counts / len(pred)

        logger.info(
            f"{np.count_nonzero(last_assignment != res['new_assignment'])}/{len(last_assignment)} assignment changes with ARI to last assignment: {adjusted_rand_score(last_assignment, res['new_assignment']):.4f}"
        )

        with np.printoptions(linewidth=np.inf):
            logger.info(f"Copy number state usage [%]:\n{100. * state_usage}")

        # NB potential conflict with GOTO logic below.
        r += 1

        if (
            # TODO config.hmrf.assignment_ari_tolerance: 0.9?
            adjusted_rand_score(res["prev_assignment"], res["new_assignment"])
            >= get_global_config().hmrf.ari_tolerance
            or len(np.unique(res["new_assignment"])) == 1  # NB single clone assigned.
            or r
            == (
                max_iter_outer - 2
            )  # NB we merge on the iteration before last, facilitating assignment to merged clones.
        ):
            if not merge:
                # NB next round we merge; and the one after fit parameters to the merged clone.
                #    skip ahead (GOTO) between iterations.
                r = max_iter_outer - 1
                merge = True

        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]

        # NB X.shape[2] is the current inferred number of clones.
        if inertia:
            log_persample_weights = np.ones((X.shape[2], n_samples)) * (
                -np.log(X.shape[2])
            )

            for sidx in range(n_samples):
                index = np.where(sample_ids == sidx)[0]

                this_persample_weight = np.bincount(
                    res["new_assignment"][index], minlength=X.shape[2]
                ) / len(index)

                log_persample_weights[:, sidx] = np.where(
                    this_persample_weight > 0, np.log(this_persample_weight), -50
                )

                log_persample_weights[:, sidx] = log_persample_weights[
                    :, sidx
                ] - scipy.special.logsumexp(log_persample_weights[:, sidx])

    # TODO FINAL
    # NB after last (merged) assignment, we calculated the clone stacks,
    #    preserve assignment keys, but update baum welch related.
    final_bm_res = pipeline_baum_welch(
        None,
        clone_stack_X,
        clone_stack_lengths,
        n_states,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
        hmmclass=hmmclass,
        params=params,
        t=t,
        random_state=random_state,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=last_log_mu,
        init_p_binom=last_p_binom,
        init_alphas=last_alphas,
        init_taus=last_taus,
        max_iter=max_iter,
        tol=tol,
        propagate_errors=propagate_hmm_param_errors,
        **remain_kwargs,
    )

    # TODO llf should also technically be updated.
    res.params = final_bm_res.params
    res.param_errors = final_bm_res.param_errors if propagate_hmm_param_errors else None
    res.profile = final_bm_res.profile

    if deconcatenate_clones:
        # NB shape=(state, segment, clone)
        res["log_gamma"] = np.stack(
            [
                res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)]
                for c in range(len(np.unique(res["new_assignment"])))
            ],
            axis=-1,
        )

        res["pred_cnv"] = np.argmax(res["log_gamma"], axis=0)

    return res


def reindex_clones(res_combine, posterior=None, single_tumor_prop=None):
    assert single_tumor_prop is None, "single_tumor_prop must be None"

    EPS_BAF = 0.05  # MAGIC
    new_res_combine = copy.copy(res_combine)

    assignments = res_combine["new_assignment"]
    clone_labels = np.unique(assignments)
    n_clones = len(clone_labels)

    pred_cnv = res_combine["pred_cnv"]

    is_concatenated = pred_cnv.ndim == 1

    if is_concatenated:
        n_obs = len(pred_cnv) // n_clones
    else:
        n_obs = pred_cnv.shape[0]

    assert res_combine["new_p_binom"].shape[1] == 1

    baf_profile_list = []
    for c in range(n_clones):
        if is_concatenated:
            clone_path = pred_cnv[c * n_obs : (c + 1) * n_obs]
        else:
            clone_path = pred_cnv[:, c]

        baf_profile_list.append(res_combine["new_p_binom"][clone_path, 0])

    baf_profiles = np.column_stack(baf_profile_list).T

    # NB normal clone minimizes deviation from 0.5 (outside the EPS_BAF deadband)
    baf_penalty = np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0)
    cid_normal = int(np.argmin(np.sum(baf_penalty, axis=1)))

    unique_clones, spot_counts = np.unique(assignments, return_counts=True)

    mask_rest = unique_clones != cid_normal
    cid_rest = unique_clones[mask_rest]
    counts_rest = spot_counts[mask_rest]

    cid_rest_sorted = cid_rest[np.argsort(counts_rest)]

    reidx = np.concatenate(([cid_normal], cid_rest_sorted)).astype(int)

    logger.info(
        f"Remapping clone index: {cid_normal} (normal) to 0, otherwise sorted by spot count."
    )

    max_id = np.max(unique_clones)
    palette = np.zeros(max_id + 1, dtype=int)

    for new_idx, old_idx in enumerate(reidx):
        palette[old_idx] = new_idx

    new_res_combine["new_assignment"] = palette[assignments]

    for key in ["new_log_mu", "new_alphas", "new_p_binom", "new_taus"]:
        if res_combine[key].shape[1] > 1:
            new_res_combine[key] = res_combine[key][:, reidx]

    if is_concatenated:
        concat_idx = np.concatenate(
            [np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx]
        )

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        if "log_gamma" in res_combine.keys():
            new_res_combine["log_gamma"] = res_combine["log_gamma"][:, concat_idx]

    else:
        if pred_cnv.shape[1] > 1:
            new_res_combine["pred_cnv"] = pred_cnv[:, reidx]

        if "log_gamma" in res_combine.keys():
            log_gamma = res_combine["log_gamma"]
            if log_gamma.ndim == 3 and log_gamma.shape[2] > 1:
                new_res_combine["log_gamma"] = log_gamma[:, :, reidx]

    if posterior is not None and posterior.shape[1] > 1:
        new_posterior = copy.copy(posterior)[:, reidx]
    else:
        new_posterior = posterior

    return new_res_combine, new_posterior

# TODO FINAL
def merge_by_minspots(
    assignment,
    res,
    single_total_bb_RD,
    min_spots_thresholds=50,
    min_umicount_thresholds=0,
    single_tumor_prop=None,
    threshold=0.5,
    adjacency_mat=None,
):
    if adjacency_mat is not None:
        raise NotImplementedError()
    else:
        logger.warning_once("TODO: adjacency_mat not queried by merge_by_minspots.")

    n_clones = len(np.unique(assignment))
    if n_clones == 1:
        merged_groups = [[assignment[0]]]
        return merged_groups, res

    # NB genomic axis is concatenated across clones.
    n_obs = int(len(res["pred_cnv"]) / n_clones)
    new_assignment = copy.copy(assignment)
    if single_tumor_prop is None:
        tmp_single_tumor_prop = np.array([1] * len(assignment))
    else:
        tmp_single_tumor_prop = single_tumor_prop

    unique_assignment = np.unique(new_assignment)

    # NB find entries in unique_assignment such that either:
    #    i) min_spots_thresholds
    #    ii) (SNP) min_umicount_thresholds are not satisfied
    # NB find clones failing min_spots_thresholds
    insufficient_spots_clones = [
        c
        for c in unique_assignment
        if np.sum(new_assignment[tmp_single_tumor_prop > threshold] == c)
        < min_spots_thresholds
    ]

    # NB find clones failing min_umicount_thresholds
    insufficient_umi_clones = [
        c
        for c in unique_assignment
        if np.sum(
            single_total_bb_RD[
                :, (new_assignment == c) & (tmp_single_tumor_prop > threshold)
            ]
        )
        < min_umicount_thresholds
    ]

    # NB log each condition separately
    logger.info(
        f"Found {len(insufficient_spots_clones)} clones with < {min_spots_thresholds} spots: {insufficient_spots_clones}"
    )
    logger.info(
        f"Found {len(insufficient_umi_clones)} clones with < {min_umicount_thresholds:_} SNP UMIs: {insufficient_umi_clones}"
    )

    # TODO
    # failed_clones = list(set(insufficient_spots_clones) | set(insufficient_umi_clones))
    failed_clones = [
        c
        for c in unique_assignment
        if (
            np.sum(new_assignment[tmp_single_tumor_prop > threshold] == c)
            < min_spots_thresholds
        )
        or (
            np.sum(
                single_total_bb_RD[
                    :, (new_assignment == c) & (tmp_single_tumor_prop > threshold)
                ]
            )
            < min_umicount_thresholds
        )
    ]
    logger.info(
        f"Found {len(failed_clones)} new clones failing thresholds on min. spots or min. SNP umis."
    )

    # NB find the remaining unique_assigment that satisfies both thresholds
    successful_clones = [c for c in unique_assignment if not c in failed_clones]

    if len(successful_clones) == 0:
        logger.error(
            f"All clones failed min. spots or min. SNP UMIs thresholds; cannot proceed with merging."
        )
        raise RuntimeError()

    # NB initial merging groups: each successful clone is its own group
    merging_groups = [[i] for i in successful_clones]

    if len(failed_clones) > 0:
        for c in failed_clones:
            # NB assigns failed clone to that with largest snp umis.
            idx_max = np.argmax(
                [
                    np.sum(
                        single_total_bb_RD[
                            :,
                            (new_assignment == c_prime)
                            & (tmp_single_tumor_prop > threshold),
                        ]
                    )
                    for c_prime in successful_clones
                ]
            )
            logger.warning(
                f"Assigning failed clone {c} to clone {[successful_clones[idx_max]]} (with largest SNP UMIs)."
            )

            merging_groups[idx_max].append(c)

    # NB re-map new_assignment according to merging_groups.
    map_clone_id = {}
    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in new_assignment])

    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.nan

    # Extract the representative clone IDs to keep
    rep_clones = [c[0] for c in merging_groups]

    # NB expect an array of clones concatenated along the genomic axis,
    if res["pred_cnv"].ndim == 1:
        n_obs = len(res["pred_cnv"]) // n_clones
        merged_res["pred_cnv"] = np.concatenate(
            [res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
        merged_res["log_gamma"] = np.hstack(
            [res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
    # NB expect an independent clone axis.
    else:
        merged_res["pred_cnv"] = res["pred_cnv"][:, rep_clones]
        merged_res["log_gamma"] = res["log_gamma"][:, :, rep_clones]

    return merging_groups, merged_res

'''
def pipeline_clone_assignment_nostack(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    res,
    pred,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    hmmclass=None,
    return_posterior=False,
):
    return pipeline_clone_assignment(
        single_X=single_X,
        single_base_nb_mean=single_base_nb_mean,
        single_total_bb_RD=single_total_bb_RD,
        res=res,
        pred=pred,
        adjacency_mat=adjacency_mat,
        prev_assignment=prev_assignment,
        sample_ids=sample_ids,
        spatial_weight=spatial_weight,
        smooth_mat=smooth_mat,
        log_persample_weights=log_persample_weights,
        single_tumor_prop=single_tumor_prop,
        hmmclass=hmmclass,
        return_posterior=return_posterior,
        merge=True,
    )
'''