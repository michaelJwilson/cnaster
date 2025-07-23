import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)

# TODO
# DEFAULT_MAX_ALLELE_COPY = 5
# DEFAULT_MAX_TOTAL_COPY = 6
# DEFAULT_MAX_MEDPLOIDY = 4
# DEFAULT_EPS_BAF = 0.05
# DEFAULT_EPS_POINTS = 0.1
# DEFAULT_MIN_PROP_THRESHOLD = 0.1

# DEFAULT_MAX_HILL_CLIMB_ITER = 10
# DEFAULT_RANDOM_RESTARTS = 20
# DEFAULT_MU_THRESHOLD = 0.3


def find_diploid_balanced_state(
    new_log_mu, new_p_binom, pred_cnv, min_prop_threshold, EPS_BAF
):
    n_states = len(new_log_mu)
    # NB find candidate diploid balanced state under the criteria that:
    #    (1) #bins in that state > 0.1 * total #bins 
    #    (2) BAF is close to 0.5 by EPS_BAF distance
    candidate = np.where(
        (
            np.bincount(pred_cnv, minlength=n_states)
            >= min_prop_threshold * len(pred_cnv)
        )
        & (np.abs(new_p_binom - 0.5) <= EPS_BAF)
    )[0]
    if len(candidate) == 0:
        raise ValueError("No candidate diploid balanced state found!")
    else:
        # NB the diploid balanced states is the one in candidate with smallest new_log_mu
        return candidate[np.argmin(new_log_mu[candidate])]


def hill_climbing_integer_copynumber_oneclone(
    new_log_mu,
    base_nb_mean,
    new_p_binom,
    pred_cnv,
    max_allele_copy=5,
    max_total_copy=6,
    max_medploidy=4,
    enforce_states={},
    EPS_BAF=0.05,
):
    n_states = len(new_log_mu)
    lambd = base_nb_mean / np.sum(base_nb_mean)
    weight_per_state = np.array([np.sum(lambd[pred_cnv == s]) for s in range(n_states)])
    mu = np.exp(new_log_mu)

    EPS_POINTS = 0.1

    def f(params, ploidy):
        # params of size (n_states, 2)
        if np.any(np.sum(params, axis=1) == 0):
            return len(pred_cnv) * 1e6
        denom = weight_per_state.dot(np.sum(params, axis=1))
        frac_rdr = np.sum(params, axis=1) / denom
        frac_baf = params[:, 0] / np.sum(params, axis=1)
        points_per_state = np.bincount(pred_cnv, minlength=params.shape[0]) + EPS_POINTS
        ### temp penalty ###
        mu_threshold = 0.3
        crucial_ordered_pairs_1 = (mu[:, None] - mu[None, :] > mu_threshold) * (
            np.sum(params, axis=1)[:, None] - np.sum(params, axis=1)[None, :] < 0
        )
        crucial_ordered_pairs_2 = (mu[:, None] - mu[None, :] < -mu_threshold) * (
            np.sum(params, axis=1)[:, None] - np.sum(params, axis=1)[None, :] > 0
        )
        # penalty on setting unbalanced states when BAF is close to 0.5
        if np.sum(params[:, 0] == params[:, 1]) > 0:
            baf_threshold = max(
                EPS_BAF,
                np.max(np.abs(new_p_binom[(params[:, 0] == params[:, 1])] - 0.5)),
            )
        else:
            baf_threshold = EPS_BAF
        unbalanced_penalty = (params[:, 0] != params[:, 1]).dot(
            np.abs(new_p_binom - 0.5) < baf_threshold
        )
        # penalty on ploidy
        derived_ploidy = np.sum(params, axis=1).dot(points_per_state) / np.sum(
            points_per_state, axis=0
        )
        return (
            np.square(0.3 * (mu - frac_rdr)).dot(points_per_state)
            + np.square(new_p_binom - frac_baf).dot(points_per_state)
            + np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            + np.sum(crucial_ordered_pairs_2) * len(pred_cnv)
            + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)
            + unbalanced_penalty * len(pred_cnv)
        )
        ### end temp penalty ###

    def hill_climb(initial_params, ploidy, max_iter=10):
        best_obj = f(initial_params, ploidy)
        params = copy.copy(initial_params)
        increased = True
        for counter in range(max_iter):
            increased = False
            for k in range(params.shape[0]):
                if k in enforce_states:
                    continue
                this_best_obj = best_obj
                this_best_k = copy.copy(params[k, :])
                for candi in candidates:
                    params[k, :] = candi
                    obj = f(params, ploidy)
                    if obj < this_best_obj:
                        this_best_obj = obj
                        this_best_k = candi
                increased = increased | (this_best_obj < best_obj)
                params[k, :] = this_best_k
                best_obj = this_best_obj
            if not increased:
                break
        return params, best_obj

    # candidate integer copy states
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )
    # find the best copy number states starting from various ploidy
    best_obj = np.inf
    best_integer_copies = np.zeros((n_states, 2), dtype=int)
    for ploidy in range(1, max_medploidy + 1):
        initial_params = np.ones((n_states, 2), dtype=int) * int(ploidy / 2)
        initial_params[:, 1] = ploidy - initial_params[:, 0]
        for k, v in enforce_states.items():
            initial_params[k] = v
        params, obj = hill_climb(initial_params, ploidy)
        if obj < best_obj:
            best_obj = obj
            best_integer_copies = copy.copy(params)
    return best_integer_copies, best_obj


def hill_climbing_integer_copynumber_fixdiploid(
    new_log_mu,
    base_nb_mean,
    new_p_binom,
    pred_cnv,
    max_allele_copy=5,
    max_total_copy=6,
    max_medploidy=4,
    min_prop_threshold=0.1,
    EPS_BAF=0.05,
    nonbalance_bafdist=None,
    nondiploid_rdrdist=None,
    enforce_states={},
):
    EPS_POINTS = 0.1

    n_states = len(new_log_mu)
    lambd = base_nb_mean / np.sum(base_nb_mean)
    weight_per_state = np.array([np.sum(lambd[pred_cnv == s]) for s in range(n_states)])
    mu = np.exp(new_log_mu)

    def is_nondiploidnormal(k):
        if not nonbalance_bafdist is None:
            if np.abs(new_p_binom[k] - 0.5) > nonbalance_bafdist:
                return True
        if not nondiploid_rdrdist is None:
            if np.abs(mu[k] - 1) > nondiploid_rdrdist:
                return True
        return False

    def f(params, ploidy, scalefactor):
        # params of size (n_states, 2)
        if np.any(np.sum(params, axis=1) == 0):
            return len(pred_cnv) * 1e6
        frac_rdr = np.sum(params, axis=1) / scalefactor
        frac_baf = params[:, 0] / np.sum(params, axis=1)
        points_per_state = np.bincount(pred_cnv, minlength=params.shape[0]) + EPS_POINTS
        ### temp penalty ###
        mu_threshold = 0.3
        crucial_ordered_pairs_1 = (mu[:, None] - mu[None, :] > mu_threshold) * (
            np.sum(params, axis=1)[:, None] - np.sum(params, axis=1)[None, :] < 0
        )
        crucial_ordered_pairs_2 = (mu[:, None] - mu[None, :] < -mu_threshold) * (
            np.sum(params, axis=1)[:, None] - np.sum(params, axis=1)[None, :] > 0
        )
        # penalty on ploidy
        derived_ploidy = np.sum(params, axis=1).dot(points_per_state) / np.sum(
            points_per_state, axis=0
        )
        return (
            np.square(0.3 * (mu - frac_rdr)).dot(points_per_state)
            + np.square(new_p_binom - frac_baf).dot(points_per_state)
            + np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            + np.sum(crucial_ordered_pairs_2) * len(pred_cnv)
            + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)
        )

    def hill_climb(initial_params, ploidy, idx_diploid_normal, max_iter=10):
        scalefactor = 2.0 / mu[idx_diploid_normal]
        best_obj = f(initial_params, ploidy, scalefactor)
        params = copy.copy(initial_params)
        increased = True
        for counter in range(max_iter):
            increased = False
            for k in range(params.shape[0]):
                if k == idx_diploid_normal or k in enforce_states:
                    continue
                this_best_obj = best_obj
                this_best_k = copy.copy(params[k, :])
                for candi in candidates:
                    if is_nondiploidnormal(k) and candi[0] == 1 and candi[1] == 1:
                        continue
                    params[k, :] = candi
                    obj = f(params, ploidy, scalefactor)
                    if obj < this_best_obj:
                        this_best_obj = obj
                        this_best_k = candi
                increased = increased | (this_best_obj < best_obj)
                params[k, :] = this_best_k
                best_obj = this_best_obj
            if not increased:
                break
        return params, best_obj

    # diploid normal state
    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=min_prop_threshold,
        EPS_BAF=EPS_BAF,
    )
    # candidate integer copy states
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )
    # find the best copy number states starting from various ploidy
    best_obj = np.inf
    best_integer_copies = np.zeros((n_states, 2), dtype=int)
    for ploidy in range(1, max_medploidy + 1):
        np.random.seed(0)

        for r in range(20):
            initial_params = candidates[
                np.random.randint(low=0, high=candidates.shape[0], size=n_states), :
            ]
            initial_params[idx_diploid_normal] = np.array([1, 1])
            for k, v in enforce_states.items():
                initial_params[k] = v
            params, obj = hill_climb(initial_params, ploidy, idx_diploid_normal)
            if obj < best_obj:
                best_obj = obj
                best_integer_copies = copy.copy(params)
    return best_integer_copies, best_obj
