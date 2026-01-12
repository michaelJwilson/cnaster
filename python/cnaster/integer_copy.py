import copy
import numpy as np
import logging
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)
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


# TODO immutable?
def get_ordered_acn():
    return (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (2, 2),
        (3, 1),
        (4, 0),
        (3, 2),
        (4, 1),
        (5, 0),
        (3, 3),
        (4, 2),
        (5, 1),
        (6, 0),
    )


def get_acn_baf_rdr(acn):
    acn = np.array(acn)
    total_copy = acn[:, 0] + acn[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        baf = np.where(total_copy > 0, acn[:, 0] / total_copy, np.nan)

    rdr = total_copy / 2.0

    return baf, rdr


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
        normal_candidate = candidate[np.argmin(new_log_mu[candidate])]

        logger.info(
            f"Found candidate normal state with new_log_mu={new_log_mu[normal_candidate]} and p_binom={new_p_binom[normal_candidate]}"
        )

        return normal_candidate


def hill_climbing_integer_copynumber_oneclone(
    new_log_mu,
    base_nb_mean,
    new_p_binom,
    pred_cnv,
    max_allele_copy=5,
    max_total_copy=6,
    max_medploidy=4,
    enforce_states={},  # MUTABLE DEFAULT
    EPS_BAF=0.05,
    expression_weight=False,
):
    n_states = len(new_log_mu)

    logger.info(f"Assuming expression weight={expression_weight}.")

    if not expression_weight:
        lambd = base_nb_mean / np.sum(base_nb_mean)
    else:
        lambd = np.ones_like(lambd) / len(lambd)

    weight_per_state = np.array([np.sum(lambd[pred_cnv == s]) for s in range(n_states)])

    logger.info(f"Found weight per state:\n{weight_per_state}")

    mu = np.exp(new_log_mu)

    EPS_POINTS = 0.1
    points_per_state = np.bincount(pred_cnv, minlength=n_states) + EPS_POINTS
    points_per_state_norm = np.sum(points_per_state, axis=0)

    config = get_global_config()
    # rdr_weight = float(config.int_copy_num.rdr_weight)
    mu_threshold = 0.3

    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=0.1,  # MAGIC
        EPS_BAF=EPS_BAF,
    )

    # scalefactor = 2.0 / mu[idx_diploid_normal]

    def f(
        params,
        ploidy,
        order_penalty=False,
        unbalanced_penalty=False,
    ):
        total_copies = np.sum(params, axis=1)

        # params of size (n_states, 2)
        if np.any(total_copies == 0):
            return len(pred_cnv) * 1e6

        # denom = weight_per_state.dot(total_copies)

        # TODO HACK
        # frac_rdr = total_copies / denom
        frac_rdr = total_copies / 2.0
        frac_baf = params[:, 0] / total_copies

        # DEPRECATE
        # NB penalty on setting unbalanced states when BAF is close to 0.5
        # if np.sum(params[:, 0] == params[:, 1]) > 0:
        #    baf_threshold = max(
        #        EPS_BAF,
        #        np.max(np.abs(new_p_binom[(params[:, 0] == params[:, 1])] - 0.5)),
        #    )
        #
        #    logger.warning(f"Assumed baf_threshold={baf_threshold} due to {np.sum(params[:, 0] == params[:, 1])} balanced param states")

        # else:
        #     baf_threshold = EPS_BAF

        derived_ploidy = total_copies.dot(points_per_state) / points_per_state_norm

        # result =  (
        #     np.square(rdr_weight * (mu - frac_rdr)).dot(points_per_state)
        #     + np.square(new_p_binom - frac_baf).dot(points_per_state)
        #     + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)
        # )

        result = np.abs(1.0 - frac_rdr / mu).dot(points_per_state) + np.abs(
            1.0 - frac_baf / new_p_binom
        ).dot(points_per_state)

        if derived_ploidy > ploidy:
            result += np.abs(1.0 - derived_ploidy / ploidy) * len(pred_cnv)

        if order_penalty:
            crucial_ordered_pairs_1 = (mu[:, None] - mu[None, :] > mu_threshold) * (
                total_copies[:, None] - total_copies[None, :] < 0
            )

            crucial_ordered_pairs_2 = (mu[:, None] - mu[None, :] < -mu_threshold) * (
                total_copies[:, None] - total_copies[None, :] > 0
            )

            result += np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            result += np.sum(crucial_ordered_pairs_2) * len(pred_cnv)

        if unbalanced_penalty:
            baf_threshold = EPS_BAF

            unbalanced_penalty = (params[:, 0] != params[:, 1]).dot(
                np.abs(new_p_binom - 0.5) < baf_threshold
            )
            result += unbalanced_penalty * len(pred_cnv)

        return result

    def hill_climb(initial_params, ploidy, max_iter=10):
        best_obj = f(initial_params, ploidy)
        params = copy.copy(initial_params)
        increased = True
        for _ in range(max_iter):
            increased = False

            # NB loop over states
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
        else:
            logger.warning(f"Reached max_iter={max_iter} on hill_climb")

        return params, best_obj

    # NB candidate integer copy states
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )

    logger.info(
        f"Solving for max_allele_copy={max_allele_copy}, max_total_copy={max_total_copy}, max ploidy={max_medploidy} for candidate states:\n{candidates}"
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

            logger.info(
                f"Found best solution with cost={best_obj:.6f} and integer copies:\n{best_integer_copies}"
            )

    logger.info(
        f"Solved for mu, p_binom, points per stat and integer copies= with best cost={best_obj:.6f}\n"
    )

    for m, p, pts, best_copy in zip(
        mu, new_p_binom, points_per_state, best_integer_copies
    ):
        logger.info(f"\t{m:7.4f}\t{p:7.4f}\t{pts:8.1f}\t{tuple(best_copy)}")

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
    enforce_states={},  # MUTABLE DEFAULT
    max_samples=20,  # MAGIC
):
    n_states = len(new_log_mu)

    EPS_POINTS = 0.1
    points_per_state = np.bincount(pred_cnv, minlength=n_states) + EPS_POINTS
    points_per_state_norm = np.sum(points_per_state, axis=0)

    mu_threshold = 0.3  # MAGIC

    mu = np.exp(new_log_mu)
    valid_ordered_mu = mu[:, None] - mu[None, :] > mu_threshold
    valid_ordered_mu_minus = mu[:, None] - mu[None, :] < -mu_threshold

    logger.info(f"Solving for mu, p_binom and points per state=\n")

    for m, p, pts in zip(mu, new_p_binom, points_per_state):
        logger.info(f"\t{m:.4f}\t{p:.4f}\t{pts:.1f}")

    def is_nondiploidnormal(k):
        """
        Check if state k is non-diploid normal under the criteria that:
        (1) BAF is away from 0.5 by nonbalance_bafdist distance
        (2) RDR is away from 1 by nondiploid_rdrdist distance
        """
        if not nonbalance_bafdist is None:
            if np.abs(new_p_binom[k] - 0.5) > nonbalance_bafdist:
                return True
        if not nondiploid_rdrdist is None:
            if np.abs(mu[k] - 1) > nondiploid_rdrdist:
                return True
        return False

    def f(params, ploidy, scalefactor):
        # NB - params of size (n_states, 2)
        #    - enforce zero copy states to have large cost
        total_copies = np.sum(params, axis=1)

        if np.any(total_copies == 0):
            return len(pred_cnv) * 1e6

        frac_rdr = total_copies / scalefactor
        frac_baf = params[:, 0] / total_copies

        # NB penalty on state pairs (i,j) where state i has higher RDR by more than mu_threshold,
        #    but lower total copy number, and vice versa.
        crucial_ordered_pairs_1 = valid_ordered_mu * (
            total_copies[:, None] - total_copies[None, :] < 0
        )
        crucial_ordered_pairs_2 = valid_ordered_mu_minus * (
            total_copies[:, None] - total_copies[None, :] > 0
        )

        # NB penalty on state ploidy weighted by points_per_state,
        #    if this is > desired ploidy (+ 0.5 in margin) it takes a cost hit,
        #    solution cost shared by all states.
        derived_ploidy = total_copies.dot(points_per_state) / points_per_state_norm

        return (
            np.square(0.3 * (mu - frac_rdr)).dot(points_per_state)  # MAGIC
            + np.square(new_p_binom - frac_baf).dot(points_per_state)
            + np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            + np.sum(crucial_ordered_pairs_2) * len(pred_cnv)
            + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)  # MAGIC
        )

    # NB python uses late binding for closures - the variable lookup happens when the function is called, not when it's defined
    def hill_climb(initial_params, ploidy, idx_diploid_normal, max_iter=10):
        # NB scaling of RDR for normal state to two copies.
        scalefactor = 2.0 / mu[idx_diploid_normal]
        best_obj = f(initial_params, ploidy, scalefactor)
        params = copy.copy(initial_params)
        increased = True

        for _ in range(max_iter):
            increased = False

            for k in range(params.shape[0]):
                # NB skip state if conditioned.
                if k == idx_diploid_normal or k in enforce_states:
                    continue

                this_best_obj = best_obj
                this_best_k = copy.copy(params[k, :])

                # TODO HACK?
                # trial_candidates = candidates if is_nondiploidnormal(k) else [[1,1]]

                for candi in candidates:
                    # NB (1,1) cannot be set to state k as real copy number is not normal.
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

            # NB stop if sweep through states yielded no improvement.
            if not increased:
                break
        else:
            logger.warning(f"Reached max_iter={max_iter} on hill_climb.")
        return params, best_obj

    # NB diploid normal state
    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=min_prop_threshold,
        EPS_BAF=EPS_BAF,
    )
    # NB candidate integer copy states
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )

    logger.info(
        f"Solving for max_allele_copy={max_allele_copy}, max_total_copy={max_total_copy}, max ploidy={max_medploidy} and max_samples={max_samples} for candidate states:\n{candidates}"
    )
    logger.info(
        f"Assuming nonbalance_bafdist={nonbalance_bafdist} and nondiploid_rdrdist={nondiploid_rdrdist} to define non-normal states."
    )

    # NB find the best copy number states starting for various ploidy
    best_obj = np.inf
    best_integer_copies = np.zeros((n_states, 2), dtype=int)

    for ploidy in range(1, max_medploidy + 1):
        np.random.seed(0)

        for _ in range(max_samples):  # MAGIC
            # DEPRECATE
            initial_params = candidates[
                np.random.randint(
                    low=0,
                    high=candidates.shape[0],
                    size=n_states,
                ),
                :,
            ]
            # non_normal_candidates = list(range(candidates.shape[0]))
            # non_normal_candidates.remove(idx_diploid_normal)

            # initial_params_idx = np.random.choice(a=non_normal_candidates, size=n_states, replace=False)
            # initial_params = candidates[initial_params_idx, :]

            initial_params[idx_diploid_normal] = np.array([1, 1])

            for k, v in enforce_states.items():
                initial_params[k] = v

            # NB sort initial_params by increasing A and B:
            # initial_params = initial_params[np.lexsort((initial_params[:, 1], initial_params[:, 0]))]

            params, obj = hill_climb(initial_params, ploidy, idx_diploid_normal)

            # logger.info(f"Solved for cost={obj:.6f} with trial solution=\n{np.hstack((initial_params, params))}")

            if obj < best_obj:
                best_obj = obj
                best_integer_copies = copy.copy(params)

                # logger.info(f"Found new best solution with ploidy={ploidy}, cost={best_obj:.6f} and integer copies:\n{best_integer_copies}")

    logger.info(
        f"Solved for mu, p_binom, points per stat and integer copies with best cost={best_obj:.6f}=\n"
    )

    for m, p, pts, best_copy in zip(
        mu, new_p_binom, points_per_state, best_integer_copies
    ):
        logger.info(f"\t{m:7.4f}\t{p:7.4f}\t{pts:8.1f}\t{tuple(best_copy)}")

    return best_integer_copies, best_obj
