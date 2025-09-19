import scipy
import logging
import numpy as np
import pandas as pd
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from statsmodels.base.model import GenericLikelihoodModel

logger = logging.getLogger(__name__)


class BAF_Binom(GenericLikelihoodModel):
    """
    Binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Total number of trials. In BAF case, this is the total number of SNP-covering UMIs.
    """
    def __init__(self, endog, exog, weights, exposure, offset, scaling, **kwargs):
        super(BAF_Binom, self).__init__(endog, exog, **kwargs)
        
        self.weights = weights
        self.exposure = exposure
        self.offset = offset
        self.scaling = scaling

    def nloglikeobs(self, params):
        linear_term = self.exog @ params
        p = self.scaling / (1 + np.exp(-linear_term + self.offset))
        llf = scipy.stats.binom.logpmf(self.endog, self.exposure, p)
        return -llf.dot(self.weights)

    # TODO fitting infrastructure
    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwargs):
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                # TODO BUG default params.
                start_params = 0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams)
        return super(BAF_Binom, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )


def identify_normal_spots(
    single_X,
    single_total_bb_RD,
    new_assignment,
    pred_cnv,
    p_binom,
    min_count,
    EPS_BAF=0.05,
    COUNT_QUANTILE=0.05,
    MIN_TOTAL=10,
):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele count per bin per spot.

    new_assignment : array, shape (n_spots,)
        Clone assignment for each spot.

    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.
    """
    # aggregate counts for each state, and evaluate the betabinomial likelihood given 0.5
    # spots with the highest likelihood are identified as normal spots
    n_obs, _, n_spots = single_X.shape

    # NB clones are stacked along genome axis.
    n_clones = int(len(pred_cnv) / n_obs)
    n_states = p_binom.shape[0]
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order="F")

    baf_profiles = p_binom[reshaped_pred_cnv, 0].T

    # NB clone closest to normal given BAF and ignoring BAF deviations < EPS_BAF.
    id_nearnormal_clone = np.argmin(
        np.sum(np.maximum(np.abs(baf_profiles - 0.5) - EPS_BAF, 0), axis=1)
    )

    umi_quantile = np.quantile(np.sum(single_X[:, 0, :], axis=0), COUNT_QUANTILE)

    baf_deviations = np.ones(n_spots)

    for i in range(n_spots):
        if (
            new_assignment[i] == id_nearnormal_clone
            and np.sum(single_X[:, 0, i])
            >= umi_quantile  # NB only consider spots as normal reference if UMIs > umi_quantile.
        ):
            # enumerate the partition of all clones to aggregate counts, and list the BAF of each partition
            this_bafs = []

            # NB each reference clone defines the genomic bins in a given state, recompute BAF by aggregating for each state and clone.
            for c in range(n_clones):
                agg_b_count = np.array(
                    [
                        np.sum(single_X[reshaped_pred_cnv[:, c] == s, 1, i])
                        for s in range(n_states)
                    ]
                )
                agg_t_count = np.array(
                    [
                        np.sum(single_total_bb_RD[reshaped_pred_cnv[:, c] == s, i])
                        for s in range(n_states)
                    ]
                )

                isin = agg_t_count >= MIN_TOTAL

                this_bafs.append(agg_b_count[isin] / agg_t_count[isin])

            this_bafs = np.concatenate(this_bafs)
            baf_deviations[i] = np.max(np.abs(this_bafs - 0.5))

    sorted_idx = np.argsort(baf_deviations)
    summed_counts = np.cumsum(np.sum(single_X[:, 0, sorted_idx], axis=0))

    # NB assign to normal the spots with smallest BAF deviation from 0.5 until enough UMI are included.
    n_normal = np.where(summed_counts >= min_count)[0][0]

    # NB boolean mask if the spots are considered normal.
    return baf_deviations <= baf_deviations[sorted_idx[n_normal]]


def identify_loh_per_clone(
    single_X,
    new_assignment,
    pred_cnv,
    p_binom,
    normal_candidate,
    single_total_bb_RD,
    MIN_SNPUMI=10,
    MAX_RDR=1,
    MIN_BAF_DEVIATION_RANGE=[0.25, 0.12],  # MUTABLE DEFAULT
    MIN_BINS_PER_STATE=10,
    MIN_BINS_ALL=25,
):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    new_assignment : array, shape (n_spots,)
        Clone assignment for each spot.

    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.

    p_binom : array, shape (n_states, 1)
        Estimated BAF per copy number state (shared across clones).

    Returns
    ----------
    loh_states : array
        An array of copy number states that are identified as LOH.

    is_B_loss : array
        A boolean array indicating whether B allele is lost (alternative A allele is lost).

    rdr_values : array
        An array of RDR values corresponding to LOH states.
    """
    n_obs = single_X.shape[0]
    n_clones = int(len(pred_cnv) / n_obs)
    n_states = p_binom.shape[0]
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order="F")

    # NB normalized RDR for (assumed) normal spots.
    simple_rdr_normal = np.sum(single_X[:, 0, (normal_candidate == True)], axis=1)
    simple_rdr_normal = simple_rdr_normal / np.sum(simple_rdr_normal)

    # NB (n_obs x n_spots) matrix for the total spot read count, distributed across bins according to the "normal" profile.
    simple_single_base_nb_mean = simple_rdr_normal.reshape(-1, 1) @ np.sum(
        single_X[:, 0, :], axis=0
    ).reshape(1, -1)

    clone_index = [np.where(new_assignment == c)[0] for c in range(n_clones)]
    X, base_nb_mean, _, _ = merge_pseudobulk_by_index_mix(
        single_X,
        simple_single_base_nb_mean,
        np.zeros(
            simple_single_base_nb_mean.shape
        ),  # NB single_total_bb_RD assumed zero.
        clone_index,
    )

    rdr_values = []
    for s in np.arange(n_states):
        rdr_values.append(
            np.sum(X[:, 0, :][reshaped_pred_cnv == s])
            / np.sum(base_nb_mean[reshaped_pred_cnv == s])
        )
    rdr_values = np.array(rdr_values)

    # NB snp-covering umi per clone.
    clone_snpumi = np.array(
        [np.sum(single_total_bb_RD[:, new_assignment == c]) for c in range(n_clones)]
    )

    # NB for each clone, sort the BAF deviations and select the largest ones (the MIN_BINS_ALL-th, e.g. 25th, largest).
    k_baf_deviation = np.sort(np.abs(p_binom[reshaped_pred_cnv, 0] - 0.5), axis=0)[
        -MIN_BINS_ALL, :
    ]
    
    # LOH states
    for threshold in np.arange(
        MIN_BAF_DEVIATION_RANGE[0], MIN_BAF_DEVIATION_RANGE[1] - 0.01, -0.02 # MAGICs
    ):
        clones_hightumor = np.where(
            (k_baf_deviation >= threshold) & (clone_snpumi >= MIN_SNPUMI * n_obs)
        )[0]
        
        if len(clones_hightumor) == 0:
            continue
        if len(clones_hightumor) == n_clones:
            clones_hightumor = np.argsort(k_baf_deviation)[1:]
            
        # LOH states
        loh_states = np.where(
            (np.abs(p_binom[:, 0] - 0.5) > threshold)
            & (np.bincount(pred_cnv, minlength=n_states) >= MIN_BINS_PER_STATE)
            & (rdr_values <= MAX_RDR)
        )[0]
        is_B_lost = p_binom[loh_states, 0] < 0.5
        if np.all(
            [
                np.sum(pd.Series(reshaped_pred_cnv[:, c]).isin(loh_states))
                >= MIN_BINS_ALL
                for c in clones_hightumor
            ]
        ):
            logger.info(
                f"Found BAF deviation threshold = {threshold} with LOH states: {loh_states} yields clones with high tumor proportion: {clones_hightumor}."
            )
            break
    else:
        logger.warning("Failed; propagating current BAF deviation threshold = {threshold} with LOH states: {loh_states} and clones with high tumor proportion: {clones_hightumor}.")

    return loh_states, is_B_lost, rdr_values[loh_states], clones_hightumor


def estimator_tumor_proportion(
    single_X,
    single_total_bb_RD,
    assignments,
    pred_cnv,
    loh_states,
    is_B_lost,
    rdr_values,
    clone_to_consider,
    smooth_mat=None,
    MIN_TOTAL=10,
):
    """
    Attributes
    ----------
    single_X : array, shape (n_obs, 2, n_spots)
        Observed transcript counts and B allele count per bin per spot.

    single_total_bb_RD : array, shape (n_obs, n_spots)
        Total allele count per bin per spot.

    assignments : pd.DataFrame of size n_spots with columns "coarse", "combined"
        Clone assignment for each spot.

    pred_cnv : array, shape (n_obs * n_clones)
        Copy number states across bins for each clone.

    loh_states, is_B_lost, rdr_values: array
        Copy number states and RDR values corresponding to LOH.

    Formula
    ----------
    0.5 ( 1. - theta ) / (theta * RDR + 1. - theta) = B_count / Total_count for each LOH state.
    """
    def estimate_purity(T_loh, B_loh, rdr_values):
        idx = np.where(T_loh > 0)[0]
        model = BAF_Binom(
            endog=B_loh[idx],
            exog=np.ones((len(idx), 1)),
            weights=np.ones(len(idx)),
            exposure=T_loh[idx],
            offset=np.log(rdr_values[idx]),
            scaling=0.5,
        )
        res = model.fit(disp=False)
        return 1.0 / (1.0 + np.exp(res.params))

    n_obs, _, n_spots = single_X.shape[0]
    n_clones = int(len(pred_cnv) / n_obs)
    reshaped_pred_cnv = pred_cnv.reshape((n_obs, n_clones), order="F")

    tumor_proportion = np.zeros(n_spots)
    full_tumor_proportion = np.zeros((n_spots, n_clones))
    
    for i in range(n_spots):
        # get adjacent spots for smoothing
        if smooth_mat is not None:
            idx_adj = smooth_mat[i, :].nonzero()[1]
        else:
            idx_adj = np.array([i])
        estimation_based_on_clones_single = np.ones(n_clones) * np.nan
        estimation_based_on_clones_smoothed = np.ones(n_clones) * np.nan
        summed_T_single = np.ones(n_clones)
        summed_T_smoothed = np.ones(n_clones)
        for c in clone_to_consider:
            # single
            B_loh = np.array(
                [
                    (
                        np.sum(single_X[:, 1, i][reshaped_pred_cnv[:, c] == s])
                        if is_B_lost[j]
                        else np.sum(
                            single_total_bb_RD[:, i][reshaped_pred_cnv[:, c] == s]
                        )
                        - np.sum(single_X[:, 1, i][reshaped_pred_cnv[:, c] == s])
                    )
                    for j, s in enumerate(loh_states)
                ]
            )
            T_loh = np.array(
                [
                    np.sum(single_total_bb_RD[:, i][reshaped_pred_cnv[:, c] == s])
                    for s in loh_states
                ]
            )
            if np.all(T_loh == 0):
                continue
            estimation_based_on_clones_single[c] = estimate_purity(
                T_loh, B_loh, rdr_values
            )
            summed_T_single[c] = np.sum(T_loh)
            # smoothed
            B_loh = np.array(
                [
                    (
                        np.sum(single_X[:, 1, idx_adj][reshaped_pred_cnv[:, c] == s])
                        if is_B_lost[j]
                        else np.sum(
                            single_total_bb_RD[:, idx_adj][reshaped_pred_cnv[:, c] == s]
                        )
                        - np.sum(single_X[:, 1, idx_adj][reshaped_pred_cnv[:, c] == s])
                    )
                    for j, s in enumerate(loh_states)
                ]
            )
            T_loh = np.array(
                [
                    np.sum(single_total_bb_RD[:, idx_adj][reshaped_pred_cnv[:, c] == s])
                    for s in loh_states
                ]
            )
            if np.all(T_loh == 0):
                continue
            estimation_based_on_clones_smoothed[c] = estimate_purity(
                T_loh, B_loh, rdr_values
            )
            summed_T_smoothed[c] = np.sum(T_loh)
        full_tumor_proportion[i, :] = estimation_based_on_clones_single
        if (assignments.combined.values[i] in clone_to_consider) and summed_T_single[
            assignments.combined.values[i]
        ] >= MIN_TOTAL:
            tumor_proportion[i] = estimation_based_on_clones_single[
                assignments.combined.values[i]
            ]
        elif (
            assignments.combined.values[i] in clone_to_consider
        ) and summed_T_smoothed[assignments.combined.values[i]] >= MIN_TOTAL:
            tumor_proportion[i] = estimation_based_on_clones_smoothed[
                assignments.combined.values[i]
            ]
        elif not assignments.combined.values[i] in clone_to_consider:
            tumor_proportion[i] = estimation_based_on_clones_single[
                np.argmax(summed_T_single)
            ]
        else:
            tumor_proportion[i] = np.nan

    tumor_proportion = np.where(tumor_proportion < 0, 0, tumor_proportion)
    return tumor_proportion, full_tumor_proportion
