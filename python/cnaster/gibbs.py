import scipy
import numpy as np
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

def gibbs_minimize(
    endog,
    exog,
    weights,
    exposure,
    nloglikeobs_func,
    bounds,
    options,
    initial_params=None,
    log_space=False,
    batch_frac=0.95,
):
    """
    # TODO (badly) needs to implement a E-step after each parameter update.

    Gibbs sampling-based minimization for EM derived emission parameters.

    Algorithm:

    1. Initialize all K state parameters to a single random draw from the empirical ratios (endog/exposure).
    2. Iterate through each state k:
       - Consider the empirical ratios for all N data points as potential updates for parameter k.
       - Calculate likelihood of model if state k adopts the parameter implied by data point i.
       - Sample new parameter for state k based on softmax of likelihoods.
    3. Update dispersion parameter after each Gibbs step (with learning rate).

    where
        endog: (N x K,) tiled array of observed counts, e.g. number of successes (BN) or counts (NB)
        exposure: (N x K,) tiled array of exposures, e.g. number of trials (BN) or baseline count (NB).
        exog: (N x K, K) one-hot encoded state indicator per datapoint.
        weights: (N x K,) array of e.g. e-step derived posterior weights P(Q_zi = k | data, params).
    """
    maxiter = int(options.get("maxiter", 100))
    
    # NB extract endog and exposure without state tiling for empirical ratio calculation by taking the data for the
    #    first state only
    isin = np.where(exog[:, 0] == 1)[0]

    single_endog = endog[isin]
    single_exposure = exposure[isin]
    
    num_states = exog.shape[1]

    # Calculate empirical parameters from data:
    # For NB: log mu = log(endog / exposure)
    # For BB: p = endog / exposure
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = single_endog / single_exposure
        valid_mask = (single_exposure > 0) & (ratios > bounds[0][0]) & (ratios < bounds[0][1])

    candidate_params = ratios[valid_mask]

    if log_space:
        candidate_params = np.log(ratios[valid_mask])

    num_candidates = len(candidate_params)

    logger.info(f"Solved for {num_candidates} Gibbs candidate parameters:\n{np.unique(candidate_params)[:5]} ... {np.unique(candidate_params)[-5:]}")
    
    logger.info(f"Initialized with relative state weights:\n{np.sum(weights, axis=0) / np.sum(weights)}")

    if initial_params is None:
        # NB initial initialization for all K states.
        current_state_params = [np.random.choice(candidate_params, size=1)[0]] * num_states
    
        # NB initial dispersion by sampling in log10 space of bounds
        disp_candidate = np.logspace(
            np.log10(bounds[-1][0]), np.log10(bounds[-1][1]), num=100
        )

        current_disp = [np.random.choice(disp_candidate)]

        current_full_params = np.array(current_state_params + current_disp)
    else:
        current_full_params = np.array(initial_params)

    # NB energy/cost to be minimized.
    best_params = current_full_params.copy()
    best_cost = -nloglikeobs_func(
        endog,
        exog,
        weights,
        exposure,
        current_full_params,
    )

    logger.info(f"Gibbs initial cost to be minimized={best_cost:.4e} @\n{best_params}")

    batch_size = int(num_candidates * batch_frac) if batch_frac is not None else num_candidates
    loss_history = []

    # NB Each iteration corresponds to {Gibbs update for each state and a dispersion update}.
    for _ in range(maxiter):
        # NB potential updates for each of the state mean parameters.
        batch_indices = np.random.choice(num_candidates, batch_size, replace=False)
        active_candidates = candidate_params[batch_indices]

        for k in range(num_states):          
            # NB ensure current value is maintained.
            sample_candidates = np.unique(np.concatenate([active_candidates, [current_full_params[k]]]))
            log_probs, valid_candidates = [], []

            for cand in sample_candidates:
                temp_params = current_full_params.copy()
                temp_params[k] = cand

                new_cost = -nloglikeobs_func(
                    endog,
                    exog,
                    weights,
                    exposure,
                    temp_params,
                )
                
                log_probs.append(-new_cost)
                valid_candidates.append(cand)
            
            log_probs = np.array(log_probs)

            # NB softmax with numerical stability.
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= np.sum(probs)
            
            # NB update state mean with sampled candidate.
            current_full_params[k] = np.random.choice(valid_candidates, p=probs)
            """
            # NB Optimize dispersion for current state configuration
            dispersion_cost = lambda disp: -nloglikeobs_func(
                endog,
                exog,
                weights,
                exposure,
                np.concatenate([current_full_params[:-1], [disp]]),
            )

            # NB constrained scalar minimization for dispersion
            res = scipy.optimize.minimize_scalar(
                dispersion_cost,
                bounds=(bounds[-1][0], bounds[-1][1]),
                method='bounded'
            )

            if res.success:
                 current_full_params[-1] = res.x
            """
            current_cost = -nloglikeobs_func(
                endog,
                exog,
                weights,
                exposure,
                current_full_params,
            )

            loss_history.append(current_cost)

            logger.info(f"Gibbs sample generated new cost={current_cost:.4e} @\n{current_full_params}")

            """            
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = current_full_params.copy()
        
                logger.info(f"Gibbs minimization found new best cost={best_cost:.4e} @\n{best_params}")
            """
            """
            # NB simple adaptive learning rate decay
            if it > 0 and it % 10 == 0:
                learning_rate *= 0.99
            """
                
    exit(0)

    result = scipy.optimize.OptimizeResult(
        x=best_params,
        fun=-best_cost,
        success=True,
        nit=maxiter,
        message="Gibbs sampling optimization terminated"
    )

    return result