import logging

import numpy as np
from numba import njit
from pysam import samples
from sklearn.mixture import GaussianMixture
from cnaster.config import get_global_config
from cnaster.hmm_emission import Weighted_BetaBinom_mix, Weighted_NegativeBinomial_mix, nloglikeobs_bb
from cnaster.hmm_update import get_em_solver_params

logger = logging.getLogger(__name__)


def get_eff_element(t, K, two_sided=True):
    result = -np.log((t * K - 1.0) / (K - 1.0))
    result = 1./result
    
    if two_sided:
        result *= 2.0
        
    return result

@njit
def interval_mean(arr, N):
    num_groups = arr.shape[0] // N

    # TODO generalize shape definition 
    result = np.empty((num_groups, *arr.shape[1:]), dtype=arr.dtype)
    
    for j in range(num_groups):
        start_idx = j * N
        end_idx = start_idx + N
        result[j, ...] = np.mean(arr[start_idx:end_idx, ...], axis=0)
    
    return result

def kmeans_plusplus(samples, k=5, scale=10.0):
    """
    kmeans++ centers (of degree k) and their associated cost.

    NB samples new center according to -log prob. of
       existing centers.
    """
    idx = np.arange(len(samples))

    # NB -log Probability.
    information = np.ones_like(samples)
    centers = []

    while len(centers) < k:
        ps = information / information.sum()

        # NB high exclusive; with replacement.
        xx = samples[np.random.choice(idx, p=ps)]
        centers.append(xx)

        # TODO
        # information = get_cost(samples, centers)

    return np.array(centers + [information.sum()])


def cna_mixture_init(
    n_states,
    t,
    X,
    base_nb_mean,
    total_bb_RD,
    hmm_class,
):
    logger.info(
        f"Initializing HMM emission with CNA Mixture++."
    )
    
    eff_element = get_eff_element(t, n_states)
    eff_element = int(np.ceil(eff_element))
    
    logger.info(f"Found effective genomic element={eff_element:.3f} for (t,K)=({t},{n_states}) and {X.shape[0]} total genomic elements")
    
    # TODO sample first state from data.
    group_idx = int(np.floor(np.random.randint(X.shape[0]) // eff_element))
    start_idx, end_idx = group_idx * eff_element, (group_idx + 1) * eff_element
    
    interval_X = X[start_idx: end_idx, ...]
    interval_base_nb_mean = base_nb_mean[start_idx: end_idx, ...]
    interval_total_bb_RD = total_bb_RD[start_idx: end_idx, ...]

    # NB true of phasing partition and assumed below (currently).
    assert X.shape[-1] == 1
    
    endog = interval_X[:,1,:].flatten()
    exposure = interval_total_bb_RD.flatten()

    print(endog / exposure)

    n_samples = len(endog)
    exog = np.ones((n_samples, 1))
    weights = np.ones(n_samples)
    
    # TODO HACK
    start_params = np.array([0.5, 1.])
    solver_params = get_em_solver_params()

    solver = Weighted_BetaBinom_mix(endog, exog, weights, exposure)
    result = solver.fit(start_params=start_params, **solver_params)

    endog = X[:,1,:].flatten()
    exposure = total_bb_RD.flatten()

    n_samples = len(endog)
    exog = np.ones((n_samples, 1))
    weights = np.ones(n_samples)
    
    nllbb = nloglikeobs_bb(
        endog, exog, weights, exposure, result.params, tumor_prop=None, reduce=False
    )

    # TODO sum nllbb across all samples in each eff_element; chose next shard according
    #     to nllbb.
    num_groups = X.shape[0] // eff_element
    group_nll_sums = np.zeros(num_groups)
    
    for group_idx in range(num_groups):
        start_idx = group_idx * eff_element
        end_idx = start_idx + eff_element
        
        # NB assumes IDD for samples in each eff_element.
        group_nll_sums[group_idx] = np.sum(nllbb[start_idx:end_idx])
    
    ps = group_nll_sums / group_nll_sums.sum()
    idx = np.random.choice(range(len(group_nll_sums)), p=ps)

    print(idx)

    start_idx = idx * eff_element
    end_idx = start_idx + eff_element

    interval_X = X[start_idx: end_idx, ...]
    interval_base_nb_mean = base_nb_mean[start_idx: end_idx, ...]
    interval_total_bb_RD = total_bb_RD[start_idx: end_idx, ...]

    endog = interval_X[:,1,:].flatten()
    exposure = interval_total_bb_RD.flatten()

    print(endog / exposure)

    n_samples = len(endog)
    exog = np.ones((n_samples, 1))
    weights = np.ones(n_samples)
    
    # TODO HACK
    start_params = np.array([0.5, 1.])
    solver_params = get_em_solver_params()

    solver = Weighted_BetaBinom_mix(endog, exog, weights, exposure)
    result = solver.fit(start_params=start_params, **solver_params)

    
def gmm_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    min_binom_prob=0.1,
    max_binom_prob=0.9,
):
    logger.info(
        f"Initializing HMM emission with Gaussian Mixture Model assuming only_minor={only_minor}."
    )

    X_gmm_rdr, X_gmm_baf = None, None

    if "m" in params:
        if in_log_space:
            X_gmm_rdr = np.vstack(
                [np.log(X[:, 0, s] / base_nb_mean[:, s]) for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            offset = np.mean(X_gmm_rdr[valid])
            normalizetomax1 = np.max(X_gmm_rdr[valid]) - np.min(X_gmm_rdr[valid])

            logger.info(
                f"Assuming log-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}"
            )
        else:
            X_gmm_rdr = np.vstack(
                [X[:, 0, s] / base_nb_mean[:, s] for s in range(X.shape[2])]
            ).T
            valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

            offset = 0

            # NB TODO? assumes X_gmm_rdr.min() = 0.
            normalizetomax1 = np.max(X_gmm_rdr[valid])

            logger.info(
                f"Assuming linear-space RDR wih offset and normalization: {offset:.4f}, {normalizetomax1:.4f}"
            )

        X_gmm_rdr = (X_gmm_rdr - offset) / normalizetomax1

    if "p" in params:
        X_gmm_baf = np.vstack(
            [X[:, 1, s] / total_bb_RD[:, s] for s in range(X.shape[2])]
        ).T

        clipped = (X_gmm_baf < min_binom_prob) | (X_gmm_baf > max_binom_prob)

        logger.warning(
            f"Clipping {np.mean(clipped):.4f} of BAF values to [{min_binom_prob}, {max_binom_prob}]."
        )

        X_gmm_baf[X_gmm_baf < min_binom_prob] = min_binom_prob
        X_gmm_baf[X_gmm_baf > max_binom_prob] = max_binom_prob

    if ("m" in params) and ("p" in params):
        X_gmm = np.hstack([X_gmm_rdr, X_gmm_baf])
    elif "m" in params:
        X_gmm = X_gmm_rdr
    elif "p" in params:
        X_gmm = X_gmm_baf

    # NB resolve NAN
    num_patched = 0

    for k in range(X_gmm.shape[1]):
        last_idx_notna = -1
        for i in range(X_gmm.shape[0]):
            if last_idx_notna >= 0 and np.isnan(X_gmm[i, k]):
                X_gmm[i, k] = X_gmm[last_idx_notna, k]
                num_patched += 1
            elif not np.isnan(X_gmm[i, k]):
                last_idx_notna = i

    logger.info(
        f"Patched {num_patched/X_gmm.shape[1]:.4f} values with NaNs in input data."
    )

    valid = np.sum(np.isnan(X_gmm), axis=1) == 0
    X_gmm = X_gmm[valid, :]

    logger.info(f"Retained {np.mean(valid)} of samples after patching.")

    max_iter = get_global_config().hmm.gmm_maxiter

    # DEPRECATE if/else.
    if random_state is None:
        gmm = GaussianMixture(n_components=n_states, max_iter=max_iter).fit(X_gmm)
    else:
        gmm = GaussianMixture(
            n_components=n_states, max_iter=max_iter, random_state=random_state
        ).fit(X_gmm)

    # TODO check? score() returns per-sample log-likelihood
    logger.info(
        f"GMM: score={gmm.score(X_gmm):.6f}, converged={gmm.converged_}, iterations={gmm.n_iter_}"
    )

    # NB cast GMM fitted parameters to HMM log_mu and p_binom parameters
    if ("m" in params) and ("p" in params):
        gmm_log_mu = (
            gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = gmm.means_[:, X.shape[2] :]

        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1. - gmm_p_binom, gmm_p_binom)

    elif "m" in params:
        gmm_log_mu = (
            gmm.means_ * normalizetomax1 + offset
            if in_log_space
            else np.log(gmm.means_[:, : X.shape[2]] * normalizetomax1 + offset)
        )
        gmm_p_binom = None

    elif "p" in params:
        gmm_log_mu = None
        gmm_p_binom = gmm.means_

        if only_minor:
            gmm_p_binom = np.where(gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom)

    if np.any(gmm_p_binom > 0.5):
        logger.warning(
            f"GMM initialized p binom > 0.5, {gmm_p_binom[gmm_p_binom > 0.5]}"
        )

    logger.info(f"Solved for GMM initialized parameters:\n{gmm_log_mu}\n{gmm_p_binom}")

    return gmm_log_mu, gmm_p_binom
