import time
import copy
import logging
import numpy as np
from cnaster.icm import icm_update
from cnaster.hmrf_utils import cast_csr
from cnaster.hmm_sitewise import hmm_sitewise

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
    spatial_weight,
    log_persample_weights=None,
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

    # NB pred is the argmax posterior by genome, concatenated by clones.
    n_clones = int(len(pred) / n_obs)
    n_states = res["new_p_binom"].shape[0]

    start_time = time.time()

    # NB likelihood for all spots and all clones.
    new_assignment = copy.copy(prev_assignment)

    single_llf = np.zeros((N, n_clones))
    posterior = np.zeros((N, n_clones))

    # NB utilize tumor mixture model?
    use_mixture = single_tumor_prop is not None

    # NB compute lambda, i.e. normalized baseline expression, for mixture model
    lambd = (
        np.sum(single_base_nb_mean, axis=1) / np.sum(single_base_nb_mean)
        if use_mixture
        else None
    )

    logger.info(
        f"Solving for emission likelihood for all clones with {hmmclass.__name__} and use_mixture={use_mixture}."
    )

    for i in range(N):
        # NB neighbor spots pooled with i.
        idx = smooth_mat[i, :].nonzero()[1]

        if use_mixture:
            # filter out NaN tumor proportions
            idx = idx[~np.isnan(single_tumor_prop[idx])]

        # TODO pooled_X updated for the aggregate per spot.
        pooled_X = np.sum(single_X[:, :, idx], axis=2, keepdims=True)
        pooled_base_nb_mean = np.sum(single_base_nb_mean[:, idx], axis=1, keepdims=True)
        pooled_total_bb_RD = np.sum(single_total_bb_RD[:, idx], axis=1, keepdims=True)

        for c in range(n_clones):
            # NB decoded copy number state for each genomic bin for this clone.
            this_pred = pred[(c * n_obs) : ((c + 1) * n_obs)]

            # TODO pre-compute weighted_tp per spot.
            if use_mixture:
                if np.sum(single_base_nb_mean[:, idx] > 0) > 0:
                    mu = np.exp(res["new_log_mu"][(this_pred % n_states), :])
                    mu /= np.sum(mu * lambd)

                    weighted_tp = (np.mean(single_tumor_prop[idx]) * mu) / (
                        np.mean(single_tumor_prop[idx]) * mu
                        + 1.0
                        - np.mean(single_tumor_prop[idx])
                    )
                else:
                    weighted_tp = np.repeat(
                        np.mean(single_tumor_prop[idx]), single_X.shape[0]
                    )

                # NB compute emission probabilities,
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom_mix(
                        pooled_X,
                        pooled_base_nb_mean,
                        res["new_log_mu"],
                        res["new_alphas"],
                        pooled_total_bb_RD,
                        res["new_p_binom"],
                        res["new_taus"],
                        np.ones((n_obs, 1)) * np.mean(single_tumor_prop[idx]),
                        weighted_tp.reshape(-1, 1), # NB cast (n_obs,) to (n_obs, 1).
                    )
                )
            else:
                tmp_log_emission_rdr, tmp_log_emission_baf = (
                    hmmclass.compute_emission_probability_nb_betabinom(
                        pooled_X,
                        pooled_base_nb_mean,
                        res["new_log_mu"],
                        res["new_alphas"],
                        pooled_total_bb_RD,
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

    logger.info(f"Solving for updated clone labels with ICM.")

    adj_list = cast_csr(adjacency_mat)

    niter = icm_update(
        single_llf,
        adj_list,
        new_assignment,
        spatial_weight,
        posterior,
        tol=0.1,
        log_persample_weights=log_persample_weights,
        sample_ids=sample_ids,
    )

    logger.info(
        f"Solved for updated clone labels in {niter} ICM iterations (took {time.time() - start_time:.2f} seconds)."
    )

    # NB compute total ln likelihood.
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
