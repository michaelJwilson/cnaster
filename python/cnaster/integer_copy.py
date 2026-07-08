import copy
import numpy as np

# from cnaster.config import get_global_config
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
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
    """
    Given allelic-copies (A, B) for each segment as an array,
    return the baf and rdr.
    """
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

    # NB candidate diploid balanced state
    candidate = np.where(
        (
            np.bincount(
                pred_cnv, minlength=n_states
            )  # count the occurences for all states
            >= min_prop_threshold * len(pred_cnv)  # threshold on fraction of occurence
        )
        & (np.abs(new_p_binom - 0.5) <= EPS_BAF)  # threshold on normal-like baf
    )[0]
    if len(candidate) == 0:
        raise ValueError("No candidate diploid balanced state found!")
    else:
        # DEPRECATE the diploid balanced states has the smallest inferred log_mu (supposedly).
        # normal_candidate_idx = np.argmin(new_log_mu[candidate]

        normal_candidate_idx = np.argmin(np.abs(1.0 - np.exp(new_log_mu[candidate])))
        normal_candidate = candidate[normal_candidate_idx]

        log_mu = new_log_mu[normal_candidate]

        logger.info(
            f"Found candidate normal state with new_log_mu={new_log_mu[normal_candidate]} and p_binom={new_p_binom[normal_candidate]}"
        )

        if np.exp(log_mu) > 1.1 or np.exp(log_mu) < 0.9:
            logger.warning(
                f"Assumed normal candidate has non-normal rdr: {np.exp(log_mu):.4f}"
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
    """
    Given the k inferred best log_mu and p_binom values from the max. likelihood calculation,
    find the best integer copy number states with hill climbing and some constraints on the
    allowed integer copy states, e.g. state ordering, ploidy and implied rdr, baf.

    Returns the best integer copy states, the best objective, and the best ploidy.
    """
    n_states = len(new_log_mu)

    logger.info(f"Assuming expression weight={expression_weight}.")

    # NB weight be (non-uniform) normal expression.
    if not expression_weight:
        lambd = base_nb_mean / np.sum(base_nb_mean)
    else:
        lambd = np.ones_like(lambd) / len(lambd)

    # NB fraction of library size from each state, assuming normal copy numbers.
    weight_per_state = np.array([np.sum(lambd[pred_cnv == s]) for s in range(n_states)])

    logger.info(f"Found weight per state:\n{weight_per_state}")

    mu = np.exp(new_log_mu)

    EPS_POINTS = 0.1

    # NB count the number of occurences of each copy state, with min. occurence (default 0.1)
    points_per_state = np.bincount(pred_cnv, minlength=n_states) + EPS_POINTS
    points_per_state_norm = np.sum(points_per_state, axis=0)

    # config = get_global_config()
    # rdr_weight = float(config.int_copy_num.rdr_weight)
    mu_threshold = 0.3  # MAGIC

    """
    # NB the inferred normal candidate state index.
    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=0.1,  # MAGIC
        EPS_BAF=EPS_BAF,
    )
    """

    # scalefactor = 2.0 / mu[idx_diploid_normal]

    # NB the assumed objective.
    def f(
        params,
        ploidy,
        order_penalty=False,
        unbalanced_penalty=False,
    ):
        # params of shape (n_states, 2)
        total_copies = np.sum(params, axis=1)

        # objective returns a large number if any total copies is zero (TBC).
        if np.any(total_copies == 0):
            return len(pred_cnv) * 1e6

        # denom = weight_per_state.dot(total_copies)

        # TODO HACK
        # frac_rdr = total_copies / denom
        frac_rdr = total_copies / 2.0
        frac_baf = params[:, 0] / total_copies

        # DEPRECATE
        # NB penalty on setting unbalanced states when baf is close to 0.5
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

        # NB L1 norm on matching the prediction based on integer copies and inferred values,
        #    weighted by state.
        result = np.abs(1.0 - frac_rdr / mu).dot(points_per_state) + np.abs(
            1.0 - frac_baf / new_p_binom
        ).dot(points_per_state)

        # NB (large) penalty on exceeding the ploidy
        if derived_ploidy > ploidy:
            result += np.abs(1.0 - derived_ploidy / ploidy) * len(pred_cnv)

        # NB integer copy numbers have a natural ordering that should be preserved.
        #    penalty if the matching does not respect this order.
        if order_penalty:
            crucial_ordered_pairs_1 = (mu[:, None] - mu[None, :] > mu_threshold) * (
                total_copies[:, None] - total_copies[None, :] < 0
            )

            crucial_ordered_pairs_2 = (mu[:, None] - mu[None, :] < -mu_threshold) * (
                total_copies[:, None] - total_copies[None, :] > 0
            )

            result += np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            result += np.sum(crucial_ordered_pairs_2) * len(pred_cnv)

        # NB penalty on A != B when inferred baf is (very) close to 0.5.
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
                # NB skip states that are "enforced".
                if k in enforce_states:
                    continue

                # NB best. objective and corresponding copy numbers.
                this_best_obj = best_obj
                this_best_k = copy.copy(params[k, :])

                # NB update the best copy numbers (in params) with a new potential candidate.
                # TODO UGH candidates in a closure scope, see below.
                for candi in candidates:
                    params[k, :] = candi
                    obj = f(params, ploidy)

                    if obj < this_best_obj:
                        this_best_obj = obj
                        this_best_k = candi

                # NB we've found a better objective in a sweep of the k best (A,B).
                increased = increased | (this_best_obj < best_obj)
                params[k, :] = this_best_k
                best_obj = this_best_obj

            # NB stop if sweep through states yielded no improvement.
            if not increased:
                break
        else:
            logger.warning(f"Reached max_iter={max_iter} on hill_climb")

        return params, best_obj

    # NB candidate integer copy states (up to max_copy, with non-zero copies,
    #    and respecting max_allele_copy).
    candidates = np.array(
        [
            [i, j]
            for i in range(max_allele_copy + 1)
            for j in range(max_allele_copy + 1)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )

    logger.info(
        f"Solving best ploidy and integer copies for max_allele_copy={max_allele_copy}, max_total_copy={max_total_copy}, max ploidy={max_medploidy} given candidate states:\n{candidates}"
    )

    # NB find the best copy number states starting from various ploidies,
    best_obj = np.inf
    best_ploidy = -1
    best_integer_copies = np.zeros((n_states, 2), dtype=int)

    for ploidy in range(1, max_medploidy + 1):
        # NB perfectly balanced (or slightly off-balance, if odd) set of allele copies where the total copy
        #    number for every single state perfectly matches the current overall ploidy.
        initial_params = np.ones((n_states, 2), dtype=int) * int(ploidy / 2)
        initial_params[:, 1] = ploidy - initial_params[:, 0]

        # NB enforce input (A,B) for given states.
        for k, v in enforce_states.items():
            initial_params[k] = v
        params, obj = hill_climb(initial_params, ploidy)

        # NB should log and return best ploidy also.
        if obj < best_obj:
            best_obj = obj
            best_ploidy = ploidy
            best_integer_copies = copy.copy(params)

            logger.info(
                f"Found best solution for ploidy={best_ploidy} with cost={best_obj:.6f} and integer copies:\n{best_integer_copies}"
            )

    logger.info(
        f"Solved for mu, p_binom, points per stat and integer copies= with best cost={best_obj:.6f}\n"
    )

    for m, p, pts, best_copy in zip(
        mu, new_p_binom, points_per_state, best_integer_copies
    ):
        logger.info(f"\t{m:7.4f}\t{p:7.4f}\t{pts:8.1f}\t{tuple(best_copy)}")

    return best_integer_copies, best_obj, best_ploidy


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
    enforce_order=False,
    rdr_relative_weight=1.0,  # MAGIC previously 0.3
    enforce_states={},  # MUTABLE DEFAULT
    max_samples=20,  # MAGIC
):
    """
    Given the k inferred best log_mu and p_binom values from the max. likelihood calculation,
    find the best integer copy number states with hill climbing and some constraints on the
    allowed integer copy states, e.g. state ordering, ploidy and implied rdr, baf.

    Enforces the "best normal" state to be (1,1) and assumes measures relative to this state.

    Returns the best integer copy states, the best objective, and the best ploidy.
    """
    n_states = len(new_log_mu)

    EPS_POINTS = 1.0e-6  # MAGIC, previously 0.1
    points_per_state = np.bincount(pred_cnv, minlength=n_states) + EPS_POINTS
    points_per_state_norm = np.sum(points_per_state, axis=0)

    mu_threshold = 0.3  # MAGIC

    mu = np.exp(new_log_mu)
    valid_ordered_mu = mu[:, None] - mu[None, :] > mu_threshold
    valid_ordered_mu_minus = mu[:, None] - mu[None, :] < -mu_threshold
    """
    logger.info(f"Solving for mu, p_binom and points per state=\n")

    for m, p, pts in zip(mu, new_p_binom, points_per_state):
        logger.info(f"\t{m:.4f}\t{p:.4f}\t{pts:.1f}")
    """

    # TODO no closure.
    def is_nondiploidnormal(k):
        """
        Check if state k (indexed into inferrred real p_binom and mu) is non-normal under
        (relatively extreme) criteria that:

        (1) BAF is away from 0.5 by nonbalance_bafdist distance (if keyword is set)
        (2) RDR is away from 1 by nondiploid_rdrdist distance (if keyword is set)

        utilized to prevent an (extreme) non-normal real state being assigned normal integer
        copies.
        """
        if nonbalance_bafdist is not None:
            if np.abs(new_p_binom[k] - 0.5) > nonbalance_bafdist:
                return True
        if nondiploid_rdrdist is not None:
            if np.abs(mu[k] - 1) > nondiploid_rdrdist:
                return True
        return False

    # TODO no closure.
    # TODO rename integer params.
    def objective(params, ploidy, scalefactor):
        # NB - params of size (n_states, 2), i.e. (mu, p) for each copy state.
        #    - enforce copy states with zero copies to have near-infinite cost.
        total_copies = np.sum(params, axis=1)

        if np.any(total_copies == 0):
            return 1e6 * len(pred_cnv)

        # NB  derive real variables from integer parameterization, (A,B).
        #
        #    "one-clone" variant assumes scalefactor=2;
        frac_rdr = total_copies / scalefactor
        frac_baf = params[:, 0] / total_copies

        # NB derived ploidy is the copy state weighted average of integer total copies,
        #
        #    penalty on state ploidy weighted by points_per_state,
        #    if this is > desired ploidy (+ 0.5 in margin) it takes a (large) cost hit,
        #    solution cost shared by all states.
        derived_ploidy = total_copies.dot(points_per_state) / points_per_state_norm

        result = (
            np.square(rdr_relative_weight * (mu - frac_rdr)).dot(
                points_per_state
            )  # MAGIC
            + np.square(new_p_binom - frac_baf).dot(points_per_state)
            + np.sum(derived_ploidy > ploidy + 0.5) * len(pred_cnv)  # MAGIC
        )

        if enforce_order:
            # NB penalty on state pairs (i,j) where state i has higher real rdr by more than mu_threshold,
            #    but lower integer total copies, and vice versa.
            crucial_ordered_pairs_1 = valid_ordered_mu * (
                total_copies[:, None] - total_copies[None, :] < 0
            )
            crucial_ordered_pairs_2 = valid_ordered_mu_minus * (
                total_copies[:, None] - total_copies[None, :] > 0
            )

            result += np.sum(crucial_ordered_pairs_1) * len(pred_cnv)
            result += np.sum(crucial_ordered_pairs_2) * len(pred_cnv)

        return result

    # NB python uses late binding for closures - the variable lookup happens when the function is called,
    #    not when it's defined
    def hill_climb(initial_params, ploidy, idx_diploid_normal, max_iter=10):
        # NB scaling of RDR for normal state to two copies.
        scalefactor = 2.0 / mu[idx_diploid_normal]
        best_obj = objective(initial_params, ploidy, scalefactor)
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

                # NB find the best of candidates for this state.
                for candi in candidates:
                    # NB (1,1) cannot be set to state k as real copy number is "not normal".
                    if is_nondiploidnormal(k) and candi[0] == 1 and candi[1] == 1:
                        continue

                    params[k, :] = candi
                    obj = objective(params, ploidy, scalefactor)

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
            for i in range(1 + max_allele_copy)
            for j in range(1 + max_allele_copy)
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
    best_ploidy = -1
    best_integer_copies = np.zeros((n_states, 2), dtype=int)

    for ploidy in range(1, max_medploidy + 1):
        # TODO HUH?
        np.random.seed(0)

        for _ in range(max_samples):  # MAGIC
            # DEPRECATE random selection from input candidates.
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

            # NB fixes diploid state as (1,1).
            initial_params[idx_diploid_normal] = np.array([1, 1])

            # NB otherwise enforce input (A,B) for given states.
            for k, v in enforce_states.items():
                initial_params[k] = v

            # NB sort initial_params by increasing A and B:
            # initial_params = initial_params[np.lexsort((initial_params[:, 1], initial_params[:, 0]))]

            params, obj = hill_climb(initial_params, ploidy, idx_diploid_normal)

            # logger.info(f"Solved for cost={obj:.6f} with trial solution=\n{np.hstack((initial_params, params))}")

            # NB improve logging.
            if obj < best_obj:
                best_obj = obj
                best_integer_copies = copy.copy(params)
                best_ploidy = ploidy
                # logger.info(f"Found new best solution with ploidy={ploidy}, cost={best_obj:.6f} and integer copies:\n{best_integer_copies}")

    logger.info(
        f"Solved for mu, p_binom, points per state and integer copies with best cost={best_obj:.6f}=\n"
    )

    for m, p, pts, best_copy in zip(
        mu, new_p_binom, points_per_state, best_integer_copies
    ):
        logger.info(f"\t{m:7.4f}\t{p:7.4f}\t{pts:8.1f}\t{tuple(best_copy.tolist())}")

    return best_integer_copies, best_obj, best_ploidy


def hill_climbing_integer_copynumber_fixdiploid_milp(
    new_log_mu,
    base_nb_mean,
    new_p_binom,
    pred_cnv,
    max_allele_copy=5,
    max_total_copy=6,
    max_medploidy=4,
    min_prop_threshold=0.0,  # MAGIC, previously 0.1
    EPS_BAF=0.05,
    nonbalance_bafdist=None,
    nondiploid_rdrdist=None,
    cost_type="L1",
    enforce_order=False,
    uniform_state_weights=False,
    rdr_relative_weight=0.3,  # TODO HACK
    enforce_states={},
    max_samples=20,  # Retained in signature for compatibility, but ignored by MILP
):
    """
    Given the k inferred best log_mu and p_binom values from the max. likelihood calculation,
    find the best integer copy number states using Mixed-Integer Linear Programming (MILP).

    Enforces the "best normal" state to be (1,1) and assumes measures relative to this state.
    Returns the globally optimal integer copy states, the best objective, and the best ploidy.
    """
    n_states = len(new_log_mu)

    EPS_POINTS = 1.0e-6

    points_per_state = (
        np.ones(n_states, dtype=float)
        if uniform_state_weights
        else 1.0 * np.bincount(pred_cnv, minlength=n_states)
    )

    points_per_state += EPS_POINTS
    points_per_state_norm = np.sum(points_per_state)

    mu_threshold = 0.3

    mu = np.exp(new_log_mu).flatten()
    new_p_binom = new_p_binom.flatten()

    valid_ordered_mu = mu[:, None] - mu[None, :] > mu_threshold
    """
    logger.info(f"Solving for mu, p_binom and points per state=\n")

    for m, p, pts in zip(mu, new_p_binom, points_per_state):
        logger.info(f"\t{m:.4f}\t{p:.4f}\t{pts:.1f}")
    """
    idx_diploid_normal = find_diploid_balanced_state(
        new_log_mu,
        new_p_binom,
        pred_cnv,
        min_prop_threshold=min_prop_threshold,
        EPS_BAF=EPS_BAF,
    )

    scalefactor = 2.0 / mu[idx_diploid_normal]

    # Generate all valid candidate copy states (excluding (0,0))
    candidates = np.array(
        [
            [i, j]
            for i in range(1 + max_allele_copy)
            for j in range(1 + max_allele_copy)
            if (not (i == 0 and j == 0)) and (i + j <= max_total_copy)
        ]
    )
    n_cand = len(candidates)

    cand_total = np.sum(candidates, axis=1)
    cand_frac_rdr = cand_total / scalefactor
    cand_frac_baf = candidates[:, 0] / cand_total

    def is_nondiploidnormal(k):
        if (
            nonbalance_bafdist is not None
            and np.abs(new_p_binom[k] - 0.5) > nonbalance_bafdist
        ):
            return True
        if nondiploid_rdrdist is not None and np.abs(mu[k] - 1.0) > nondiploid_rdrdist:
            return True
        return False

    W = np.zeros((n_states, n_cand))
    for k in range(n_states):
        w_k = points_per_state[k]

        if cost_type == "L1":
            rdr_cost = rdr_relative_weight * np.abs(mu[k] - cand_frac_rdr)
            baf_cost = np.abs(new_p_binom[k] - cand_frac_baf)
        else:
            rdr_cost = rdr_relative_weight * (mu[k] - cand_frac_rdr) ** 2
            baf_cost = (new_p_binom[k] - cand_frac_baf) ** 2

        W[k, :] = w_k * (rdr_cost + baf_cost)

    c_vec = W.flatten()

    # ---------------------------------------------------------
    # 2. Build Bounds (Enforcing invalid states to 0)
    # ---------------------------------------------------------
    # Decision variables x_{k,c} are binary [0, 1]
    lb = np.zeros(n_states * n_cand)
    ub = np.ones(n_states * n_cand)

    for k in range(n_states):
        for c, cand in enumerate(candidates):
            idx = k * n_cand + c

            # Constraint: Forced Diploid State
            if k == idx_diploid_normal:
                if cand[0] != 1 or cand[1] != 1:
                    ub[idx] = 0

            # Constraint: Enforced user states
            elif k in enforce_states:
                if cand[0] != enforce_states[k][0] or cand[1] != enforce_states[k][1]:
                    ub[idx] = 0

            # Constraint: non-diploid normal filter
            elif is_nondiploidnormal(k) and cand[0] == 1 and cand[1] == 1:
                ub[idx] = 0

    bounds = Bounds(lb, ub)
    integrality = np.ones(n_states * n_cand)  # 1 indicates integer constraint

    # ---------------------------------------------------------
    # 3. Build Linear Constraints (A_matrix * x [<=, ==, >=] b_vec)
    # ---------------------------------------------------------
    constraints_A = []
    constraints_lb = []
    constraints_ub = []

    # A. One-Candidate-Per-State Constraint: sum_c x_{k,c} == 1
    A_eq = lil_matrix((n_states, n_states * n_cand))
    for k in range(n_states):
        A_eq[k, k * n_cand : (k + 1) * n_cand] = 1

    constraints_A.append(A_eq)
    constraints_lb.append(np.ones(n_states))
    constraints_ub.append(np.ones(n_states))

    # B. Ordering Constraints
    if enforce_order:
        # If mu[i] - mu[j] > mu_threshold, total_copies[i] >= total_copies[j]
        # -> sum_c (cand_total * x_{j,c}) - sum_c (cand_total * x_{i,c}) <= 0
        pairs = np.argwhere(valid_ordered_mu)
        if len(pairs) > 0:
            A_ord = lil_matrix((len(pairs), n_states * n_cand))
            for row, (i, j) in enumerate(pairs):
                A_ord[row, j * n_cand : (j + 1) * n_cand] = cand_total
                A_ord[row, i * n_cand : (i + 1) * n_cand] = -cand_total

            constraints_A.append(A_ord)
            constraints_lb.append(np.full(len(pairs), -np.inf))
            constraints_ub.append(np.zeros(len(pairs)))

    logger.info(
        f"Solving MILP for max_allele_copy={max_allele_copy}, max_total_copy={max_total_copy}, max ploidy={max_medploidy}"
    )

    best_obj = np.inf
    best_ploidy = -1
    best_integer_copies = np.zeros((n_states, 2), dtype=int)

    # ---------------------------------------------------------
    # 4. Solve across Plodies
    # ---------------------------------------------------------
    # We loop over ploidy limits just as the hill climber did, finding the globally
    # optimal assignment that satisfies the ploidy bound.
    for target_ploidy in range(1, max_medploidy + 1):

        # C. Ploidy Constraint: sum_k sum_c (w_k * cand_total * x_{k,c}) <= (target_ploidy + 0.5) * sum(w_k)
        A_ploidy = lil_matrix((1, n_states * n_cand))
        for k in range(n_states):
            A_ploidy[0, k * n_cand : (k + 1) * n_cand] = (
                points_per_state[k] * cand_total
            )

        ploidy_limit = (target_ploidy + 0.5) * points_per_state_norm

        # Combine base constraints with the current ploidy constraint
        A_mat = np.vstack([A.toarray() for A in constraints_A] + [A_ploidy.toarray()])
        lb_vec = np.concatenate(constraints_lb + [[-np.inf]])
        ub_vec = np.concatenate(constraints_ub + [[ploidy_limit]])

        lin_const = LinearConstraint(A_mat, lb_vec, ub_vec)

        # Run MILP Solver
        res = milp(
            c=c_vec,
            integrality=integrality,
            bounds=bounds,
            constraints=lin_const,
            options={"disp": False},
        )

        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_ploidy = target_ploidy

            # Extract the assigned candidates from the binary decision vector
            x_sol = np.round(res.x).astype(int)
            for k in range(n_states):
                # Find which candidate c was chosen (where x_{k,c} == 1)
                chosen_c = np.argmax(x_sol[k * n_cand : (k + 1) * n_cand])
                best_integer_copies[k] = candidates[chosen_c]

    logger.info(
        f"MILP solved for mu, p_binom, points per state and integer copies with best cost={best_obj:.6f}=\n"
    )

    for m, p, pts, best_copy in zip(
        mu, new_p_binom, points_per_state, best_integer_copies
    ):
        logger.info(f"\t{m:7.4f}\t{p:7.4f}\t{pts:8.1f}\t{tuple(best_copy.tolist())}")

    return best_integer_copies, best_obj, best_ploidy
