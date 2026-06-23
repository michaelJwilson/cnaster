import copy
import logging
import time

import numpy as np
import pandas as pd
import scipy.special
from numba import njit, prange
from pathlib import Path
from cnaster.icm import (
    icm_sweep,
    icm_sweep_deque,
    wolff_sweep,
    unpack_adjacency,
    merge_assignment,
)
from cnaster.hmm import gmm_init, pipeline_baum_welch
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.hmrf_utils import cast_csr, clone_stack_obs
from cnaster.utils import count_calls, get_output_dir, write_fig
from cnaster.hmm_initialize import plot_cna_mixture, cna_mixture_init
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.config import get_global_config
from cnaster.plotting import plot_clones_spatial
from cnaster.plot_genomic import plot_clones_genomic, plot_clones_genomic_raw
from cnaster.deprecated.hmrf import (
    aggr_hmrfmix_reassignment_concatenate as dep_aggr_hmrfmix_reassignment_concatenate,
)
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
    expected = np.arange(len(unique_ids))

    if not np.array_equal(unique_ids, expected):
        logger.error(f"Found invalid clone ids (e.g. not contiguous): {unique_ids}.")
        raise RuntimeError()

    return True


@njit(cache=True)
def pool_hmrf_data(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    smooth_indices,
    smooth_indptr,
    single_tumor_prop=None,
    use_mixture=False,
    res_new_log_mu=None,
    pred=None,
    n_states=None,
    lambd=None,
):
    """
    Precompute pooled data for all spots using neighbor aggregation.

    Parameters
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        BAF and RD count matrix for all bins in all spots.
    single_base_nb_mean : array, shape (n_obs, n_spots)
        Diploid baseline of gene expression matrix.
    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele UMI count matrix.
    smooth_adj : list of lists
        Adjacency list from cast_csr(smooth_mat), where each element is [(col, val), ...].
    single_tumor_prop : array, shape (n_spots,), optional
        Tumor proportion for each spot.
    use_mixture : bool
        Whether to use mixture model.
    res_new_log_mu : array, shape (n_states, n_clones), optional
        Log mu parameters for mixture model.
    pred : array, shape (n_obs * n_clones,), optional
        Predicted states for mixture model.
    n_states : int, optional
        Number of states for mixture model.
    lambd : array, shape (n_obs,), optional
        Lambda values for mixture model.

    Returns
    -------
    pooled_X : array, shape (n_obs, 2, n_spots)
        Pooled X data for each spot.
    pooled_base_nb_mean : array, shape (n_obs, n_spots)
        Pooled base nb mean for each spot.
    pooled_total_bb_RD : array, shape (n_obs, n_spots)
        Pooled total bb RD for each spot.
    weighted_tp : array, shape (n_obs, n_spots)
        Weighted tumor proportions for each spot (if use_mixture=True).
    mean_tumor_prop : array, shape (n_spots,)
        Mean tumor proportions for each spot.
    """
    # NB no logger comments in jit compiled.
    n_obs, n_comp, N = single_X.shape

    pooled_X = np.zeros((n_obs, n_comp, N), dtype=single_X.dtype)
    pooled_base_nb_mean = np.zeros((n_obs, N), dtype=single_base_nb_mean.dtype)
    pooled_total_bb_RD = np.zeros((n_obs, N), dtype=single_total_bb_RD.dtype)
    mean_tumor_prop, weighted_tp = None, None
    """
    # TODO HACK BUG  
    if use_mixture:
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
        start_idx = smooth_indptr[i]
        end_idx = smooth_indptr[i + 1]

        valid_neighbors = []

        for k in range(start_idx, end_idx):
            col = smooth_indices[k]

            if use_mixture and single_tumor_prop is not None:
                if not np.isnan(single_tumor_prop[col]):
                    valid_neighbors.append(col)
            else:
                valid_neighbors.append(col)

        valid_count = len(valid_neighbors)

        # TODO CHECK prior behavior?
        if valid_count == 0:
            continue

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
        # TODO HACK BUG
        if use_mixture:
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


@njit(parallel=True, cache=True)
def compute_single_llf(
    N,
    smooth_indices,
    smooth_indptr,
    nz_nb_base,
    nz_bb_total,
    single_tumor_prop,
    use_mixture,
    tmp_log_emission_rdr,
    tmp_log_emission_baf,
    pred,
    n_obs,
    n_clones,
):
    single_llf = np.zeros((N, n_clones))

    for i in prange(N):
        start_idx = smooth_indptr[i]
        end_idx = smooth_indptr[i + 1]

        sum_nb_base, sum_bb_total = 0, 0

        # NB loop over pooled neighbors of spot i,
        #    skipping those with nan tumor proportion.
        for k in range(start_idx, end_idx):
            neighbor = smooth_indices[k]

            if use_mixture:
                if np.isnan(single_tumor_prop[neighbor]):
                    continue

            sum_nb_base += nz_nb_base[neighbor]
            sum_bb_total += nz_bb_total[neighbor]

        ratio_nonzeros = 1.0

        # NB both normal and baf signals available.
        if sum_nb_base > 0 and sum_bb_total > 0:
            ratio_nonzeros = sum_bb_total / sum_nb_base

        for c in range(n_clones):
            offset = c * n_obs

            term_rdr, term_baf = 0.0, 0.0

            for o in range(n_obs):
                state = pred[offset + o]

                term_rdr += tmp_log_emission_rdr[state, o, i]
                term_baf += tmp_log_emission_baf[state, o, i]

            single_llf[i, c] = ratio_nonzeros * term_rdr + term_baf

    return single_llf


# NB aggregate by smooth mat. with tumor/normal mix, spot reassignment, concatenated by clone?
def aggr_hmrfmix_reassignment_concatenate(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    pred,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    spatial_weight,
    log_persample_weights=None,
    single_tumor_prop=None,
    hmmclass=hmm_sitewise,
    return_posterior=False,
    merge=False,
):
    n_obs, _, N = single_X.shape

    # NB pred is the argmax posterior by genome, concatenated across clones.
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]

    start_time = time.time()

    # NB clone assignment for all spots.
    new_assignment = copy.copy(prev_assignment)

    # NB utilize tumor mixture model?
    use_mixture = single_tumor_prop is not None

    # NB compute lambda, i.e. normalized baseline expression, for mixture model
    lambd = (
        np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
        if use_mixture
        else None  # TODO BUG?
    )

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={single_X.shape}, n_states={n_states} and {n_clones} clones with {hmmclass.__name__}, use_mixture={use_mixture} and merge={merge}."
    )

    logger.info("Pooling hmrf data by smooth mat. (reduces necessary computation).")

    # NB pool data by smooth mat: reduces spots to calculate likelihood for, i.e. faster.
    pooled_X, pooled_base_nb_mean, pooled_total_bb_RD, _, weighted_tp = pool_hmrf_data(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        smooth_mat.indices,
        smooth_mat.indptr,
        single_tumor_prop,
        use_mixture,
        res["new_log_mu"] if use_mixture else None,
        pred if use_mixture else None,
        n_states if use_mixture else None,
        lambd,
    )

    # NB emission shape: (n_states, n_obs, n_spots)
    if use_mixture:
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
    # NB number of non-zero nb_base and bb_total per spot.
    nz_nb_base = (single_base_nb_mean > 0).sum(axis=0)
    nz_bb_total = (single_total_bb_RD > 0).sum(axis=0)

    _tumor_prop = single_tumor_prop if single_tumor_prop is not None else np.empty(0)

    single_llf = compute_single_llf(
        N,
        smooth_mat.indices,
        smooth_mat.indptr,
        nz_nb_base,
        nz_bb_total,
        _tumor_prop,
        use_mixture,
        tmp_log_emission_rdr,
        tmp_log_emission_baf,
        pred,
        n_obs,
        n_clones,
    )

    # assert np.allclose(single_llf, new_single_llf), "BUG: single_llf mismatch"

    adj_list = cast_csr(adjacency_mat)
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

    # NB Posterior probabilities if return_posterior=True.
    posterior = np.zeros((N, n_clones))

    if get_global_config().hmrf.fixed_assignment:
        logger.warning(f"Assuming a fixed clone assignment")
    else:
        logger.info(f"Solving for updated clone labels.")

        # NB updates new_assignment and posterior in place given log emission likelihood.
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
        niter, new_cost = wolff_sweep(                                                                                                                                                   
            single_llf,                                                                                                                                                                  
            adj_spots,                                                                                                                                                                   
            adj_neighbors,                                                                                                                                                              
            adj_weights,                                                                                                                                                                
            new_assignment,                                                                                                                                                              
            spatial_weight,                                                                                                                                                             
            posterior,                                                                                                                                                                   
            log_persample_weights=log_persample_weights,                                                                                                                                 
            sample_ids=sample_ids,                                                                                                                                                       
        )   
        """
        logger.info(f"Ready for potential merging with merge={merge}.")

        while merge:
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

            if best_merge_cost > new_cost:
                u, v = best_merge_pair
                num_merged_spots = 0

                for i in range(len(new_assignment)):
                    if new_assignment[i] == u:
                        new_assignment[i] = v
                        num_merged_spots += 1

                logger.info(
                    f"Merged {num_merged_spots} spots from clone {u} into clone {v} with dC={best_merge_cost - new_cost:.6e}"
                )
                new_cost = best_merge_cost
            else:
                logger.info(
                    f"No more beneficial merges available (latest dC={best_merge_cost - new_cost:.6e})."
                )
                break

        _, cnts = np.unique(new_assignment, return_counts=True)

        logger.info(
            f"Solved for updated clone labels with new cost {new_cost:.6e} in {niter} iterations (took {time.time() - start_time:.2f} seconds with clone breakdown=\n{[f'{xx:.3f}' for xx in cnts / cnts.sum()]})."
        )

    logger.info(f"Computing total ln likelihood.")

    total_llf = np.sum(single_llf[np.arange(N), new_assignment])

    # TODO?
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )

    """
    # TODO HACK?  e.g. pred of HMM requires an clone ordering definition.
    # NB reindex new_assignment to contiguous clone ids.
    unique_ids = np.unique(new_assignment)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    new_assignment = np.array([id_map[x] for x in new_assignment])
    """
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf


def validation_summary(
    lengths,
    X,
    base_nb_mean,
    total_bb_RD,
    tumor_prop,
):
    n_segments, _, n_bulk = X.shape
    n_contigs = len(lengths)

    config = get_global_config()
    secondary_min_umi = config.quality.secondary_min_umi

    # TODO
    assert n_contigs == 22

    zero_point = 0

    for ii, ll in enumerate(lengths):
        contig_num_extreme_major_baf, contig_num_extreme_minor_baf = [], []
        contig_num_insufficient_snp_umi, contig_num_insufficient_umi = [], []

        for c in range(n_bulk):
            contig_rdrs = (
                X[zero_point : zero_point + ll, 0, c]
                / base_nb_mean[zero_point : zero_point + ll, c]
            )
            contig_bafs = (
                X[zero_point : zero_point + ll, 1, c]
                / total_bb_RD[zero_point : zero_point + ll, c]
            )

            contig_num_extreme_major_baf.append(np.count_nonzero(contig_bafs >= 0.65))
            contig_num_extreme_minor_baf.append(np.count_nonzero(contig_bafs <= 0.35))

            contig_num_insufficient_snp_umi.append(
                np.count_nonzero(
                    total_bb_RD[zero_point : zero_point + ll, c] < secondary_min_umi
                )
            )

            contig_num_insufficient_umi.append(
                np.count_nonzero(
                    base_nb_mean[zero_point : zero_point + ll, c]
                    < 10 * secondary_min_umi
                )
            )

        logger.info(
            f"Contig {1 + ii} \t {contig_num_extreme_major_baf} \t {contig_num_extreme_minor_baf} \t {contig_num_insufficient_snp_umi} \t {contig_num_insufficient_umi}"
        )

        zero_point += ll


@count_calls
def hmrfmix_concatenate_pipeline(
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
    hmmclass=hmm_sitewise,
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
    plot_progress=True,
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
        new_init_log_mu, new_init_p_binom = gmm_init(
            n_states,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            params,
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
        logger.info(
            f"Plotting initial copy state mixture for instance {hmrfmix_concatenate_pipeline.call_count-1} with X.shape={X.shape}."
        )

        n_states = init_p_binom.shape[0]

        """
        plot_cna_mixture(
            (
                np.tile(init_log_mu, n_clones).reshape(n_states, n_clones)
                if init_log_mu is not None
                else None
            ),
            (
                np.tile(init_alphas, n_clones).reshape(n_states, n_clones)
                if init_alphas is not None
                else None
            ),
            (
                np.tile(init_p_binom, n_clones).reshape(n_states, n_clones)
                if init_p_binom is not None
                else None
            ),
            (
                np.tile(init_taus, n_clones).reshape(n_states, n_clones)
                if init_taus is not None
                else None
            ),
            X,
            base_nb_mean,
            total_bb_RD,
            width=10,
            prefix=f"instance{hmrfmix_concatenate_pipeline.call_count-1}",
        )
        """
        """
        plot_cna_mixture(
            init_log_mu,
            init_alphas,
            init_p_binom,
            init_taus,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            width=10,
            prefix=f"instance{hmrfmix_concatenate_pipeline.call_count-1}_clone",
        )
        """

    last_log_mu = init_log_mu if "m" in params else None
    last_p_binom = init_p_binom if "p" in params else None
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)

    for c, idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # NB inertia to spot clone change.
    inertia = bool(get_global_config().hmrf.inertia)
    log_persample_weights = (
        np.ones((n_clones, n_samples)) * (-np.log(n_clones)) if inertia else None
    )

    logger.info(f"Assuming hmrf inertia={inertia} and {hmmclass.__name__} instance.")

    # NB required for remain_kwargs construction.
    res = {}
    r = 0

    # NB convoluted loop logic to achieve merge on last iteration.
    while r <= max_iter_outer:
        logger.info(
            f"----****  Solving iteration {r}/{max_iter_outer} of copy number state fitting & clone assignment (HMM + HMRF) ****----"
        )

        # NB segments for each clone stacked.
        sample_length = np.ones(X.shape[2], dtype=int) * X.shape[0]
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

        # NB MAP copy state, no phasing.
        pred = np.argmax(res["log_gamma"], axis=0)

        # NB TODO 'max' clone assignment.
        new_assignment, single_llf, total_llf = aggr_hmrfmix_reassignment_concatenate(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            res,
            pred,
            smooth_mat,
            adjacency_mat,
            last_assignment,
            sample_ids,
            spatial_weight=spatial_weight,
            log_persample_weights=log_persample_weights,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
            merge=merge,
        )

        # NB handle the case where one clone has zero spots.
        if len(np.unique(new_assignment)) < X.shape[2]:
            res["assignment_before_reindex"] = new_assignment
            remaining_clones = np.sort(np.unique(new_assignment))

            # NB map original clone id to new enumeration.
            re_indexing = {c: i for i, c in enumerate(remaining_clones)}

            logger.warning(
                f"Iteration {r}: detected clone loss:  re-indexing clones with map={re_indexing}"
            )

            # NB re-index new_assignment to be consecutive given a missing clone.
            new_assignment = np.array([re_indexing[x] for x in new_assignment])

            concat_idx = np.concatenate(
                [np.arange(c * n_obs, c * n_obs + n_obs) for c in remaining_clones]
            )

            # NB log_gamma and pred_cnv ordered by clone.
            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]

        res["prev_assignment"] = last_assignment
        res["new_assignment"] = new_assignment
        res["total_llf"] = total_llf

        clone_index = [
            np.where(res["new_assignment"] == c)[0]
            for c in np.sort(np.unique(res["new_assignment"]))
        ]

        X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            clone_index,
            single_tumor_prop,
            threshold=tumorprop_threshold,
        )

        # DEPRECATE? TODO.
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

        if plot_progress:
            logger.info(f"Plotting progress for interation {r}.")

            output_dir = get_output_dir()
            progress_dir = f"{output_dir}/plots/progress/"

            if not (pprogress_dir := Path(progress_dir)).exists():
                logger.info(f"Creating {progress_dir}")
                pprogress_dir.mkdir(exist_ok=True)

            # TODO HACK
            assignment = pd.Series([f"clone {x}" for x in res["new_assignment"]])
            clones_fig = plot_clones_spatial(
                coords,
                assignment,
                single_tumor_prop=single_tumor_prop,
                sample_list=sample_list,
                sample_ids=sample_ids,
                base_width=4,
                base_height=3,
            )

            fig_path = f"{progress_dir}/{prefix}_spatial_iter{r}.pdf"
            write_fig(fig_path, clones_fig, transparent=True, bbox_inches="tight")

            # TODO copy rename.
            clones_genomic = plot_clones_genomic_raw(
                single_X,
                single_base_nb_mean,
                single_total_bb_RD,
                [
                    np.where(res["new_assignment"] == c)[0]
                    for c in np.sort(np.unique(res["new_assignment"]))
                ],
                lengths,
                res=res,
                single_tumor_prop=None,
                sample_list=sample_list,
                remove_xticks=True,
                rdr_ylim=6,
                chrtext_shift=-0.2,
                base_height=3.2,
                pointsize=5,
                linewidth=1,
            )

            fig_path = f"{progress_dir}/{prefix}_genomic_iter{r}.pdf"
            write_fig(fig_path, clones_genomic, transparent=True, bbox_inches="tight")

        # NB potential conflict with GOTO logic below.
        r += 1

        if (
            # TODO config.hmrf.assignment_ari_tolerance: 0.9?
            adjusted_rand_score(last_assignment, res["new_assignment"])
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

    return res


def reindex_clones(res_combine, posterior=None, single_tumor_prop=None):
    EPS_BAF = 0.05  # MAGIC

    n_spots = len(res_combine["new_assignment"])
    n_states, n_clones = res_combine["new_p_binom"].shape

    # NB assumes not concatenated
    n_obs = res_combine["pred_cnv"].shape[0]
    new_res_combine = copy.copy(res_combine)

    if single_tumor_prop is None:
        # NB select 'near-normal' clone and set to clone 0
        pred_cnv = res_combine["pred_cnv"]
        baf_profiles = np.array(
            [res_combine["new_p_binom"][pred_cnv[:, c], c] for c in range(n_clones)]
        )
        cid_normal = np.argmin(
            np.sum(np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1)
        )

        # TODO HACK WARN discrepant clone ids [c for c in range(n_clones).
        cid_rest = np.array(
            [c for c in np.unique(res_combine["new_assignment"]) if c != cid_normal]
        ).astype(int)
        reidx = np.append(cid_normal, cid_rest)
        map_reidx = {cid: i for i, cid in enumerate(reidx)}

        logger.info(
            f"Remapping clone index according to {map_reidx}, with {cid_normal} assumed normal."
        )

        # NB re-order entries in res_combine
        new_res_combine["new_assignment"] = np.array(
            [map_reidx[c] for c in res_combine["new_assignment"]]
        )
        new_res_combine["new_log_mu"] = res_combine["new_log_mu"][:, reidx]
        new_res_combine["new_alphas"] = res_combine["new_alphas"][:, reidx]
        new_res_combine["new_p_binom"] = res_combine["new_p_binom"][:, reidx]
        new_res_combine["new_taus"] = res_combine["new_taus"][:, reidx]
        new_res_combine["log_gamma"] = res_combine["log_gamma"][:, :, reidx]
        new_res_combine["pred_cnv"] = res_combine["pred_cnv"][:, reidx]

        if posterior is not None:
            new_posterior = copy.copy(posterior)
            new_posterior = new_posterior[:, reidx]
        else:
            new_posterior = None
    else:
        # LEGACY BUG?
        raise RuntimeError()

        # NB add normal clone as clone 0
        new_res_combine["new_assignment"] = new_res_combine["new_assignment"] + 1
        new_res_combine["new_log_mu"] = np.hstack(
            [np.zeros((n_states, 1)), res_combine["new_log_mu"]]
        )
        new_res_combine["new_alphas"] = np.hstack(
            [np.zeros((n_states, 1)), res_combine["new_alphas"]]
        )
        new_res_combine["new_p_binom"] = np.hstack(
            [0.5 * np.ones((n_states, 1)), res_combine["new_p_binom"]]
        )
        new_res_combine["new_taus"] = np.hstack(
            [np.zeros((n_states, 1)), res_combine["new_taus"]]
        )
        new_res_combine["log_gamma"] = np.dstack(
            [np.zeros((n_states, n_obs, 1)), res_combine["log_gamma"]]
        )
        new_res_combine["pred_cnv"] = np.hstack(
            [np.zeros((n_obs, 1), dtype=int), res_combine["pred_cnv"]]
        )
        new_posterior = np.hstack([np.ones((n_spots, 1)) * np.nan, posterior])
    return new_res_combine, new_posterior


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
            # NB assigns failed clone to that with large SNP UMIs.
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
    merged_res["pred_cnv"] = np.concatenate(
        [
            res["pred_cnv"][(c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )
    merged_res["log_gamma"] = np.hstack(
        [
            res["log_gamma"][:, (c[0] * n_obs) : (c[0] * n_obs + n_obs)]
            for c in merging_groups
        ]
    )
    return merging_groups, merged_res


# NB point={aggr_hmrf_reassignment, aggr_hmrfmix_reassignment};
#    posterior={hmrf_reassignment_posterior;; hmrfmix_reassignment_posterior}
def aggr_hmrf_reassignment(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    pred,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    hmmclass=hmm_sitewise,
    return_posterior=False,
    merge=True,
):
    # LEGACY single ICM move per spot, i.e. likely to be ill converged.
    #        assumes parameters are a stack across clones.
    n_spots = single_X.shape[2]
    n_obs = single_X.shape[0]

    # NB clone stack of emission states.
    n_clones = res["new_log_mu"].shape[1]
    # n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((n_spots, n_clones))
    new_assignment = copy.copy(prev_assignment)

    posterior = np.zeros((n_spots, n_clones))

    logger.info("Computing unary likelihood for HMRF reassignment.")

    # TODO UGH FINAL takes forever to run.
    for i in range(n_spots):
        idx = smooth_mat[i, :].nonzero()[1]

        for c in range(n_clones):
            (
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"][:, c : (c + 1)],
                res["new_alphas"][:, c : (c + 1)],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"][:, c : (c + 1)],
                res["new_taus"][:, c : (c + 1)],
            )
            if (
                np.sum(single_base_nb_mean[:, idx] > 0) > 0
                and np.sum(single_total_bb_RD[:, idx] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, idx] > 0)
                    / np.sum(single_base_nb_mean[:, idx] > 0)
                )

                single_llf[i, c] = ratio_nonzeros * np.sum(
                    tmp_log_emission_rdr[pred[:, c], np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[pred[:, c], np.arange(n_obs), 0])
            else:
                single_llf[i, c] = np.sum(
                    tmp_log_emission_rdr[pred[:, c], np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[pred[:, c], np.arange(n_obs), 0])

        # w_node = single_llf[i, :]
        # w_node += log_persample_weights[:, sample_ids[i]]
        # w_edge = np.zeros(n_clones)
        # for j in adjacency_mat[i, :].nonzero()[1]:
        #     if new_assignment[j] >= 0:
        #         w_edge[new_assignment[j]] += adjacency_mat[i, j]
        # new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        # posterior[i, :] = np.exp(
        #     w_node
        #   + spatial_weight * w_edge
        #   - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        # )

    adj_list = cast_csr(adjacency_mat)
    adj_spots, adj_neighbors, adj_weights = unpack_adjacency(adj_list)

    # NB Posterior probabilities if return_posterior=True.
    posterior = np.zeros((n_spots, n_clones))

    if get_global_config().hmrf.fixed_assignment:
        logger.warning(f"Assuming a fixed clone assignment")
    else:
        logger.info(f"Solving for updated clone labels.")

        # NB updates new_assignment and posterior in place given log emission likelihood.
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

        while merge:
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

            if best_merge_cost > new_cost:
                num_clones = len(np.unique(new_assignment))

                if num_clones <= 2:
                    logger.warning(
                        "Found beneficial merge of final two clones; ignoring."
                    )
                    break

                u, v = best_merge_pair
                num_merged_spots = 0

                for i in range(len(new_assignment)):
                    if new_assignment[i] == u:
                        new_assignment[i] = v
                        num_merged_spots += 1

                logger.info(
                    f"Merged {num_merged_spots} spots from clone {u} into clone {v} with dC={best_merge_cost - new_cost:.6e}"
                )
                new_cost = best_merge_cost
            else:
                logger.info(
                    f"Exhausted beneficial merges (latest dC={best_merge_cost - new_cost:.6e})."
                )
                break

    # TODO UGH FINAL takes forever to run.
    # NB compute total log likelihood: log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(n_spots), new_assignment])
    for i in range(n_spots):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf

# DEPRECATE?
def hmrf_reassignment_posterior(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    hmmclass=hmm_sitewise,
    return_posterior=False,
):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    posterior = np.zeros((N, n_clones))

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]

        for c in range(n_clones):
            (
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"][:, c : (c + 1)],
                res["new_alphas"][:, c : (c + 1)],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"][:, c : (c + 1)],
                res["new_taus"][:, c : (c + 1)],
            )
            if (
                np.sum(single_base_nb_mean[:, idx] > 0) > 0
                and np.sum(single_total_bb_RD[:, idx] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, i : (i + 1)] > 0)
                    / np.sum(single_base_nb_mean[:, i : (i + 1)] > 0)
                )

                single_llf[i, c] = ratio_nonzeros * np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                )
            else:
                single_llf[i, c] = np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                )

        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            if new_assignment[j] >= 0:
                w_edge[new_assignment[j]] += adjacency_mat[i, j]
        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    # NB compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf


def aggr_hmrfmix_reassignment(
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
    hmmclass=hmm_sitewise,
    return_posterior=False,
):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    posterior = np.zeros((N, n_clones))

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            if np.sum(single_base_nb_mean, axis=1) > 0:
                mu = np.exp(res["new_log_mu"][(pred % n_states), :]) / np.sum(
                    np.exp(res["new_log_mu"][(pred % n_states), :]) * lambd
                )
                weighted_tp = (np.mean(single_tumor_prop[idx]) * mu) / (
                    np.mean(single_tumor_prop[idx]) * mu
                    + 1
                    - np.mean(single_tumor_prop[idx])
                )
            else:
                weighted_tp = np.repeat(
                    np.mean(single_tumor_prop[idx]), single_X.shape[0]
                )
            (
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"][:, c : (c + 1)],
                res["new_alphas"][:, c : (c + 1)],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"][:, c : (c + 1)],
                res["new_taus"][:, c : (c + 1)],
                np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                weighted_tp.reshape(-1, 1),
            )
            if (
                np.sum(single_base_nb_mean[:, idx] > 0) > 0
                and np.sum(single_total_bb_RD[:, idx] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, idx] > 0)
                    / np.sum(single_base_nb_mean[:, idx] > 0)
                )

                single_llf[i, c] = ratio_nonzeros * np.sum(
                    tmp_log_emission_rdr[pred[:, c], np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[pred[:, c], np.arange(n_obs), 0])
            else:
                single_llf[i, c] = np.sum(
                    tmp_log_emission_rdr[pred[:, c], np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[pred[:, c], np.arange(n_obs), 0])

        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            if new_assignment[j] >= 0:
                w_edge[new_assignment[j]] += adjacency_mat[i, j]
        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    # NB compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf


def hmrfmix_reassignment_posterior(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    res,
    smooth_mat,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    hmmclass=hmm_sitewise,
    return_posterior=False,
):
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = res["new_log_mu"].shape[1]
    n_states = res["new_p_binom"].shape[0]
    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)

    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    posterior = np.zeros((N, n_clones))

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]
        idx = idx[~np.isnan(single_tumor_prop[idx])]
        for c in range(n_clones):
            if np.sum(single_base_nb_mean) > 0:
                this_pred_cnv = res["pred_cnv"][:, c]
                logmu_shift = np.array(
                    scipy.special.logsumexp(
                        res["new_log_mu"][this_pred_cnv, c] + np.log(lambd), axis=0
                    )
                )
                kwargs = {
                    "logmu_shift": logmu_shift.reshape(1, 1),
                    "sample_length": np.array([n_obs]),
                }
            else:
                kwargs = {}
            (
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
            ) = hmmclass.compute_emission_probability_nb_betabinom_mix(
                np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                res["new_log_mu"][:, c : (c + 1)],
                res["new_alphas"][:, c : (c + 1)],
                np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                res["new_p_binom"][:, c : (c + 1)],
                res["new_taus"][:, c : (c + 1)],
                np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                **kwargs,
            )
            if (
                np.sum(single_base_nb_mean[:, idx] > 0) > 0
                and np.sum(single_total_bb_RD[:, idx] > 0) > 0
            ):
                ratio_nonzeros = (
                    1.0
                    * np.sum(single_total_bb_RD[:, i : (i + 1)] > 0)
                    / np.sum(single_base_nb_mean[:, i : (i + 1)] > 0)
                )

                single_llf[i, c] = ratio_nonzeros * np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                )
            else:
                single_llf[i, c] = np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_rdr[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                ) + np.sum(
                    scipy.special.logsumexp(
                        tmp_log_emission_baf[:, :, 0] + res["log_gamma"][:, :, c],
                        axis=0,
                    )
                )

        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            if new_assignment[j] >= 0:
                w_edge[new_assignment[j]] += adjacency_mat[i, j]
        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    # NB compute total log likelihood log P(X | Z) + log P(Z)
    total_llf = np.sum(single_llf[np.arange(N), new_assignment])
    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )
    if return_posterior:
        return new_assignment, single_llf, total_llf, posterior
    else:
        return new_assignment, single_llf, total_llf
