import copy
import logging

import numpy as np
import scipy.special
from cnaster.hmm import initialization_by_gmm, pipeline_baum_welch
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
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]

    single_llf = np.zeros((N, n_clones))
    new_assignment = copy.copy(prev_assignment)
    posterior = np.zeros((N, n_clones))

    # Determine if we're using mixture model
    use_mixture = single_tumor_prop is not None

    if use_mixture:
        # Compute lambda for mixture model
        lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

    for i in range(N):
        idx = smooth_mat[i, :].nonzero()[1]

        if use_mixture:
            # Filter out NaN tumor proportions
            idx = idx[~np.isnan(single_tumor_prop[idx])]

        for c in range(n_clones):
            this_pred = pred[(c * n_obs) : (c * n_obs + n_obs)]

            if use_mixture:
                # Compute weighted tumor proportion for mixture model
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

                # Compute emission probabilities for mixture model
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

            # Compute log likelihood (common logic)
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

        # Compute node and edge weights (common logic)
        w_node = single_llf[i, :]
        w_node += log_persample_weights[:, sample_ids[i]]

        w_edge = np.zeros(n_clones)
        for j in adjacency_mat[i, :].nonzero()[1]:
            w_edge[new_assignment[j]] += adjacency_mat[i, j]

        new_assignment[i] = np.argmax(w_node + spatial_weight * w_edge)

        # Compute posterior probabilities
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
    lambd = np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)

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
        init_log_mu, init_p_binom = initialization_by_gmm(
            n_states,
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            params,
            random_state=random_state,
            in_log_space=False,
            only_minor=False,
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

        # NB 'max' clone assignment.
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
