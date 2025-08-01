import copy
import logging

import numpy as np
import scipy.special
from cnaster.hmm import gmm_init, pipeline_baum_welch
from cnaster.hmm_sitewise import hmm_sitewise
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


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
    log_persample_weights,
    spatial_weight,
    single_tumor_prop=None,
    hmmclass=hmm_sitewise,
    return_posterior=False,
):
    """
    HMRF assign spots to tumor clones with optional mixture modeling.

    Parameters
    ----------
    single_X : array, shape (n_bins, 2, n_spots)
        BAF and RD count matrix for all bins in all spots.

    single_base_nb_mean : array, shape (n_bins, n_spots)
        Diploid baseline of gene expression matrix.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele UMI count matrix.

    res : dictionary
        Dictionary of estimated HMM parameters.

    pred : array, shape (n_bins * n_clones)
        HMM states for all bins and all clones. (Derived from forward-backward algorithm)

    smooth_mat : array, shape (n_spots, n_spots)
        Matrix used for feature propagation for computing log likelihood.

    adjacency_mat : array, shape (n_spots, n_spots)
        Adjacency matrix used to evaluate label consistency in HMRF.

    prev_assignment : array, shape (n_spots,)
        Clone assignment of the previous iteration.

    spatial_weight : float
        Scaling factor for HMRF label consistency between adjacent spots.

    single_tumor_prop : array, shape (n_spots,), optional
        Tumor proportion for each spot. If provided, uses mixture model.

    hmmclass : class, default=hmm_sitewise
        HMM class with emission probability computation methods.

    return_posterior : bool, default=False
        Whether to return posterior probabilities.

    Returns
    -------
    new_assignment : array, shape (n_spots,)
        Clone assignment of this new iteration.

    single_llf : array, shape (n_spots, n_clones)
        Log likelihood of each spot given that its label is each clone.

    total_llf : float
        The HMRF objective, which is the sum of log likelihood under the optimal labels plus the sum of edge potentials.

    posterior : array, shape (n_spots, n_clones), optional
        Posterior probabilities if return_posterior=True.
    """
    n_obs, _, N = single_X.shape
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]

    # NB likelihood for all spots and all clones.
    new_assignment = copy.copy(prev_assignment)

    single_llf = np.zeros((N, n_clones))
    posterior = np.zeros((N, n_clones))

    # NB utilize tumor mixture model?
    use_mixture = single_tumor_prop is not None

    if use_mixture:
        # NB compute lambda, i.e. normalized baseline expression, for mixture model
        lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]

        if use_mixture:
            # filter out NaN tumor proportions
            idx = idx[~np.isnan(single_tumor_prop[idx])]

        for c in range(n_clones):
            # NB copy number state for each genomic bin for this clone.
            this_pred = pred[(c * n_obs) : (c * n_obs + n_obs)]

            if use_mixture:
                # NB compute weighted tumor proportion for mixture model
                if np.sum(single_base_nb_mean[:, idx] > 0) > 0:
                    mu = np.exp(res["new_log_mu"][(this_pred % n_states), :]) / np.sum(
                        np.exp(res["new_log_mu"][(this_pred % n_states), :]) * lambd
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

                # NB compute emission probabilities, 
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom_mix(
                        np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                        np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                        res["new_log_mu"],
                        res["new_alphas"],
                        np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                        res["new_p_binom"],
                        res["new_taus"],
                        np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                        weighted_tp.reshape(-1, 1),
                    )
                )
            else:
                # Compute emission probabilities for standard model
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom(
                        np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                        np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                        res["new_log_mu"],
                        res["new_alphas"],
                        np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                        res["new_p_binom"],
                        res["new_taus"],
                    )
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
                    tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])
            else:
                single_llf[i, c] = np.sum(
                    tmp_log_emission_rdr[this_pred, np.arange(n_obs), 0]
                ) + np.sum(tmp_log_emission_baf[this_pred, np.arange(n_obs), 0])

        # NB compute node and edge weights (common logic)
        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]

        w_edge = np.zeros(n_clones)
        
        for j in adjacency_mat[i, :].nonzero()[1]:
            w_edge[new_assignment[j]] += adjacency_mat[i, j]

        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        # NB compute posterior probabilities
        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    # Compute total log likelihood (common logic)
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


def clone_stack_obs(
    X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
):
    # NB vertical stacking of X, base_nb_mean, total_bb_RD, tumor_prop across clones,
    # i.e. reshape observation data from (n_obs, 2, n_clones) to (n_obs * n_clones, 2, 1)
    clone_stack_X = np.vstack(
        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
    ).T.reshape(-1, 2, 1)

    # NB vertical stacking by clone, cast to column.
    clone_stack_base_nb_mean = base_nb_mean.flatten("F").reshape(-1, 1)
    clone_stack_total_bb_RD = total_bb_RD.flatten("F").reshape(-1, 1)

    # NB replicate lengths N clone times, as derived from X - clone num. may change.
    clone_stack_lengths = np.tile(lengths, X.shape[2])
    clone_stack_sitewise_transmat = np.tile(log_sitewise_transmat, X.shape[2])

    # NB per-clone tumor prop. repeated num_obs times.
    stack_tumor_prop = (
        np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
        if tumor_prop is not None
        else None
    )

    return (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_lengths,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
    )


def hmrfmix_concatenate_pipeline(
    outdir,
    prefix,
    single_X,
    lengths,
    single_base_nb_mean,
    single_total_bb_RD,
    single_tumor_prop,
    initial_clone_index,
    n_states,
    log_sitewise_transmat,
    coords=None,
    smooth_mat=None,
    adjacency_mat=None,
    sample_ids=None,
    max_iter_outer=5,
    nodepotential="max",
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
    unit_xsquared=9,
    unit_ysquared=3,
    spatial_weight=1.0 / 6.0,
    tumorprop_threshold=0.5,
):
    n_obs, _, n_spots = single_X.shape
    n_clones = len(initial_clone_index)

    # NB map sample_ids to integer enum.
    unique_sample_ids = np.unique(sample_ids)
    n_samples = len(unique_sample_ids)

    tmp_map_index = {unique_sample_ids[i]: i for i in range(len(unique_sample_ids))}
    sample_ids = np.array([tmp_map_index[x] for x in sample_ids])

    # TODO BUG? baseline expression by summing over all clones; relative to genome-wide.
    # NB not applicable for BAF only.
    norm = np.sum(single_base_nb_mean)

    # DEPRECATE
    assert np.isscalar(norm)

    if norm == 0.0:
        logger.warning(f"Found nb_mean=0 across all spots,segments.")

    with np.errstate(divide="ignore", invalid="ignore"):
        lambd = np.sum(single_base_nb_mean, axis=1) / norm

    X, base_nb_mean, total_bb_RD, tumor_prop = merge_pseudobulk_by_index_mix(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        initial_clone_index,
        single_tumor_prop,
        threshold=tumorprop_threshold,
    )

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

    if (init_log_mu is None) or (init_p_binom is None):
        init_log_mu, init_p_binom = gmm_init(
            n_states,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=False,  # TODO BUG?
            only_minor=False,  # TODO BUG?
        )

    last_log_mu = init_log_mu if "m" in params else None
    last_p_binom = init_p_binom if "p" in params else None
    last_alphas = init_alphas
    last_taus = init_taus
    last_assignment = np.zeros(single_X.shape[2], dtype=int)

    for c, idx in enumerate(initial_clone_index):
        last_assignment[idx] = c

    # NB inertia to spot clone change.
    log_persample_weights = np.ones((n_clones, n_samples)) * (-np.log(n_clones))

    res = {}

    for r in range(max_iter_outer):
        print("\n\n")

        logger.info(
            f"----****  Solving iteration {r}/{max_iter_outer} of copy number state fitting & clone assignment (HMM + HMRF) ****----"
        )

        # NB [num_obs for each clone / sample].
        sample_length = np.ones(X.shape[2], dtype=int) * X.shape[0]

        remain_kwargs = {"sample_length": sample_length, "lambd": lambd}

        if "log_gamma" in res:
            remain_kwargs["log_gamma"] = res["log_gamma"]

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
            log_persample_weights,
            spatial_weight=spatial_weight,
            single_tumor_prop=single_tumor_prop,
            hmmclass=hmmclass,
        )

        # NB handle the case when one clone has zero spots.
        if len(np.unique(new_assignment)) < X.shape[2]:
            logger.warning(
                "Clone %d has no spots assigned. Re-indexing clones.",
                r,
                np.unique(new_assignment),
            )

            res["assignment_before_reindex"] = new_assignment
            remaining_clones = np.sort(np.unique(new_assignment))
            re_indexing = {c: i for i, c in enumerate(remaining_clones)}

            # NB re-index new_assignment to be consecutive given a missing clone.
            new_assignment = np.array([re_indexing[x] for x in new_assignment])

            concat_idx = np.concatenate(
                [np.arange(c * n_obs, c * n_obs + n_obs) for c in remaining_clones]
            )

            res["log_gamma"] = res["log_gamma"][:, concat_idx]
            res["pred_cnv"] = res["pred_cnv"][concat_idx]

        # NB add to results.
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

        # NB max not mean.
        logger.info(
            f"ARI to last assignment: {adjusted_rand_score(last_assignment, res["new_assignment"]):.4f}"
        )

        with np.printoptions(linewidth=np.inf):
            logger.info(f"Copy number state usage [%]:\n{100. * state_usage}")

        if (
            # TODO config.hmrf.assignment_ari_tolerance: 0.9?
            adjusted_rand_score(last_assignment, res["new_assignment"]) > 0.99
            or len(np.unique(res["new_assignment"])) == 1
        ):
            break

        last_log_mu = res["new_log_mu"]
        last_p_binom = res["new_p_binom"]
        last_alphas = res["new_alphas"]
        last_taus = res["new_taus"]
        last_assignment = res["new_assignment"]

        log_persample_weights = np.ones((X.shape[2], n_samples)) * (-np.log(X.shape[2]))

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


def reindex_clones(res_combine, posterior, single_tumor_prop):
    EPS_BAF = 0.05  # MAGIC
    n_spots = posterior.shape[0]
    n_obs = res_combine["pred_cnv"].shape[0]
    n_states, n_clones = res_combine["new_p_binom"].shape
    new_res_combine = copy.copy(res_combine)
    new_posterior = copy.copy(posterior)

    if single_tumor_prop is None:
        # NB select 'near-normal' clone and set to clone 0
        pred_cnv = res_combine["pred_cnv"]
        baf_profiles = np.array(
            [res_combine["new_p_binom"][pred_cnv[:, c], c] for c in range(n_clones)]
        )
        cid_normal = np.argmin(
            np.sum(np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1)
        )
        cid_rest = np.array([c for c in range(n_clones) if c != cid_normal]).astype(int)
        reidx = np.append(cid_normal, cid_rest)
        map_reidx = {cid: i for i, cid in enumerate(reidx)}
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
        new_posterior = new_posterior[:, reidx]
    else:
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
):
    n_clones = len(np.unique(assignment))
    if n_clones == 1:
        merged_groups = [[assignment[0]]]
        return merged_groups, res

    n_obs = int(len(res["pred_cnv"]) / n_clones)
    new_assignment = copy.copy(assignment)
    if single_tumor_prop is None:
        tmp_single_tumor_prop = np.array([1] * len(assignment))
    else:
        tmp_single_tumor_prop = single_tumor_prop
    unique_assignment = np.unique(new_assignment)
    # NB find entries in unique_assignment such that either: i) min_spots_thresholds ii) min_umicount_thresholds are not satisfied
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
        f"Found {len(failed_clones)} new clones failing thresholds on min. spots or min. umis."
    )

    # NB find the remaining unique_assigment that satisfies both thresholds
    successful_clones = [c for c in unique_assignment if not c in failed_clones]
    # NB initial merging groups: each successful clone is its own group
    merging_groups = [[i] for i in successful_clones]
    # NB for each failed clone, assign them to the 'closest' successful clone
    if len(failed_clones) > 0:
        for c in failed_clones:
            # NB selects new clone with largest RD amongst those viable.
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
            merging_groups[idx_max].append(c)
    map_clone_id = {}
    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in new_assignment])

    merged_res = copy.copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.NAN
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


"""
def compute_hmrf_assignment_likelihood(
    single_llf,
    adjacency_mat,
    prev_assignment,
    sample_ids,
    log_persample_weights,
    spatial_weight,
    return_posterior=False,
):
    N, n_clones = single_llf.shape
    new_assignment = copy.copy(prev_assignment)
    posterior = np.zeros((N, n_clones))

    for i in range(N):
        # Node potential
        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]

        # Edge potential
        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            if new_assignment[j] >= 0:
                w_edge[new_assignment[j]] += adjacency_mat[i, j]

        # Assignment
        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        # Posterior
        posterior[i, :] = np.exp(
            w_node
            + spatial_weight * w_edge
            - scipy.special.logsumexp(w_node + spatial_weight * w_edge)
        )

    total_llf = np.sum(single_llf[np.arange(N), new_assignment])

    for i in range(N):
        total_llf += np.sum(
            spatial_weight
            * np.sum(
                new_assignment[adjacency_mat[i, :].nonzero()[1]] == new_assignment[i]
            )
        )

    if return_posterior:
        return new_assignment, total_llf, posterior
    else:
        return new_assignment, total_llf


def _compute_likelihood_matrix(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    smooth_mat,
    pred=None,
    single_tumor_prop=None,
    hmmclass=hmm_sitewise,
    use_posterior=False,
):
    n_obs, _, N = single_X.shape
    n_states = res["new_p_binom"].shape[0]
    n_clones = res["new_log_mu"].shape[1]

    single_llf = np.zeros((N, n_clones))

    use_mixture = single_tumor_prop is not None

    if use_mixture:
        lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]

        if use_mixture:
            idx = idx[~np.isnan(single_tumor_prop[idx])]

        for c in range(n_clones):
            if use_mixture:
                emission_args = _get_mixture_emission_args(
                    single_X,
                    single_base_nb_mean,
                    single_total_bb_RD,
                    res,
                    idx,
                    c,
                    single_tumor_prop,
                    pred,
                    lambd,
                    n_obs,
                    n_states,
                    use_posterior,
                )
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom_mix(
                        *emission_args
                    )
                )
            else:
                emission_args = _get_standard_emission_args(
                    single_X, single_base_nb_mean, single_total_bb_RD, res, idx, c
                )
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom(*emission_args)
                )

            single_llf[i, c] = _compute_spot_likelihood(
                tmp_log_emission_rdr,
                tmp_log_emission_baf,
                single_base_nb_mean,
                single_total_bb_RD,
                idx,
                i,
                pred,
                res,
                c,
                n_obs,
                use_posterior,
            )

    return single_llf


def _get_standard_emission_args(
    single_X, single_base_nb_mean, single_total_bb_RD, res, idx, c
):
    return (
        np.sum(single_X[:, :, idx], axis=2, keepdims=True),
        np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
        res["new_log_mu"][:, c : (c + 1)],
        res["new_alphas"][:, c : (c + 1)],
        np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
        res["new_p_binom"][:, c : (c + 1)],
        res["new_taus"][:, c : (c + 1)],
    )


def _get_mixture_emission_args(
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    res,
    idx,
    c,
    single_tumor_prop,
    pred,
    lambd,
    n_obs,
    n_states,
    use_posterior,
):
    base_args = _get_standard_emission_args(
        single_X, single_base_nb_mean, single_total_bb_RD, res, idx, c
    )

    tumor_prop_arg = np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx])

    if use_posterior:
        # Posterior-based mixture
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
        return base_args + (tumor_prop_arg,) + tuple(kwargs.values())
    else:
        # Point estimate mixture
        if np.sum(single_base_nb_mean[:, idx] > 0) > 0:
            mu = np.exp(res["new_log_mu"][(pred % n_states), :]) / np.sum(
                np.exp(res["new_log_mu"][(pred % n_states), :]) * lambd
            )
            weighted_tp = (np.mean(single_tumor_prop[idx]) * mu) / (
                np.mean(single_tumor_prop[idx]) * mu
                + 1
                - np.mean(single_tumor_prop[idx])
            )
        else:
            weighted_tp = np.repeat(np.mean(single_tumor_prop[idx]), n_obs)

        return base_args + (tumor_prop_arg, weighted_tp.reshape(-1, 1))
"""


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
            tmp_log_emission_rdr, tmp_log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom(
                    np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                    np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                    res["new_log_mu"][:, c : (c + 1)],
                    res["new_alphas"][:, c : (c + 1)],
                    np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                    res["new_p_binom"][:, c : (c + 1)],
                    res["new_taus"][:, c : (c + 1)],
                )
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

    # NB compute total log likelihood: log P(X | Z) + log P(Z)
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
            tmp_log_emission_rdr, tmp_log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom(
                    np.sum(single_X[:, :, idx], axis=2, keepdims=True),
                    np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True),
                    res["new_log_mu"][:, c : (c + 1)],
                    res["new_alphas"][:, c : (c + 1)],
                    np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True),
                    res["new_p_binom"][:, c : (c + 1)],
                    res["new_taus"][:, c : (c + 1)],
                )
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
            if np.sum(single_base_nb_mean[:, idx] > 0) > 0:
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
            tmp_log_emission_rdr, tmp_log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
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
            tmp_log_emission_rdr, tmp_log_emission_baf = (
                hmmclass.compute_emission_probability_nb_betabinom_mix(
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
