import numpy as np
import scipy
from numba import njit
from cnaster.count_encoder import CountEncoder
from cnaster.config import get_global_config
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# NB define global alias for legacy
construct_unique_matrix = CountEncoder.construct_unique_encoding


def get_em_solver_params():
    """
    Get the parameters for the emission solver.
    """
    config = get_global_config()
    solver = config.hmm.solver

    match solver:
        case "BFGS":
            kwargs = ("xrtol", "disp")
        case "L-BFGS-B":
            kwargs = ("maxiter", "ftol", "disp")
        case "Nelder-Mead":
            kwargs = ("maxiter", "xtol", "ftol", "disp")
        case _:
            raise ValueError(f"cnaster does not support solver: {solver}")

    return {k.replace("em_", ""): float(getattr(config.hmm, f"em_{k}")) for k in kwargs}


@njit
def mylogsumexp(a):
    a_max = np.max(a)

    if np.isinf(a_max):
        return a_max

    tmp = np.exp(a - a_max)

    s = np.sum(tmp)
    s = np.log(s)

    return s + a_max


@njit
def np_max_ax_keep(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros((1, arr.shape[1]))
        for i in range(result.shape[1]):
            result[:, i] = np.max(arr[:, i])
    else:
        result = np.zeros((arr.shape[0], 1))
        for i in range(result.shape[0]):
            result[i, :] = np.max(arr[i, :])
    return result


@njit
def np_sum_ax_keep(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.zeros((1, arr.shape[1]))
        for i in range(result.shape[1]):
            result[:, i] = np.sum(arr[:, i])
    else:
        result = np.zeros((arr.shape[0], 1))
        for i in range(result.shape[0]):
            result[i, :] = np.sum(arr[i, :])
    return result


@njit
def mylogsumexp_ax_keep(a, axis):
    a_max = np_max_ax_keep(a, axis=axis)

    tmp = np.exp(a - a_max)

    s = np_sum_ax_keep(tmp, axis=axis)
    s = np.log(s)

    return s + a_max


@njit
def np_sum_ax_squeeze(arr, axis=0):
    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.zeros(arr.shape[1])

        for i in range(len(result)):
            result[i] = np.sum(arr[:, i])
    else:
        result = np.empty(arr.shape[0])

        for i in range(len(result)):
            result[i] = np.sum(arr[i, :])

    return result


"""
def convert_params(mean, std):
    p = mean / std**2
    n = mean * p / (1.0 - p)

    return n, p
"""


@njit
def convert_params_disp(mean, overdisp):
    p = 1.0 / (1.0 + overdisp * mean)

    # NB guard on min. overdispersion, such that (overdisp * mean) << 1.
    n = 1.0 / np.maximum(overdisp, 1.0e-10)

    return n, p


def calc_sparsity(csr_matrix):
    total_elements = csr_matrix.shape[0] * csr_matrix.shape[1]
    non_zero_elements = csr_matrix.size

    return (total_elements - non_zero_elements) / total_elements


def compute_posterior_obs(log_alpha, log_beta):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
    Output:
        log_gamma: size n_states * n_observations. gamma[i,t] = P(q_t = i | O, lambda). gamma[i, t] propto alpha[i,t] * beta[i,t]
    """
    n_states = log_alpha.shape[0]
    n_obs = log_alpha.shape[1]

    # log_gamma = np.zeros((n_states, n_obs))
    log_gamma = log_alpha + log_beta

    if np.any(np.sum(log_gamma, axis=0) == 0):
        logger.error("Sum of posterior probability is zero for some observations!")
        raise RuntimeError()

    log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)

    return log_gamma


@njit
def compute_posterior_transition_sitewise(
    log_alpha, log_beta, log_transmat, log_emission
):
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]

    log_xi = np.zeros((2 * n_states, 2 * n_states, n_obs - 1))

    for i in np.arange(2 * n_states):
        for j in np.arange(2 * n_states):
            for t in np.arange(n_obs - 1):
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = (
                    log_alpha[i, t]
                    + log_transmat[
                        i - n_states * int(i / n_states),
                        j - n_states * int(j / n_states),
                    ]
                    + np.sum(log_emission[j, t + 1, :])
                    + log_beta[j, t + 1]
                )

    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])

    return log_xi


@njit
def compute_posterior_transition_nophasing(
    log_alpha, log_beta, log_transmat, log_emission
):
    """
    Input
        log_alpha: output from forward_lattice_gaussian. size n_states * n_observations. alpha[j, t] = P(o_1, ... o_t, q_t = j | lambda).
        log_beta: output from backward_lattice_gaussian. size n_states * n_observations. beta[i, t] = P(o_{t+1}, ..., o_T | q_t = i, lambda).
        log_transmat: n_states * n_states. Transition probability after log transformation.
        log_emission: n_states * n_observations * n_spots. Log probability.
    Output:
        log_xi: size n_states * n_states * (n_observations-1). xi[i,j,t] = P(q_t=i, q_{t+1}=j | O, lambda)
    """
    n_states = int(log_alpha.shape[0] / 2)
    n_obs = log_alpha.shape[1]
    # initialize log_xi
    log_xi = np.zeros((n_states, n_states, n_obs - 1))
    # compute log_xi
    for i in np.arange(n_states):
        for j in np.arange(n_states):
            for t in np.arange(n_obs - 1):
                # ??? Theoretically, joint distribution across spots under iid is the prod (or sum) of individual (log) probabilities.
                # But adding too many spots may lead to a higher weight of the emission rather then transition prob.
                log_xi[i, j, t] = (
                    log_alpha[i, t]
                    + log_transmat[i, j]
                    + np.sum(log_emission[j, t + 1, :])
                    + log_beta[j, t + 1]
                )
    # normalize
    for t in np.arange(n_obs - 1):
        log_xi[:, :, t] -= mylogsumexp(log_xi[:, :, t])
    return log_xi


def get_solver():
    known_solvers = ("BFGS", "L-BFGS-B", "Nelder-Mead")

    name = get_global_config().hmm.solver

    assert (
        name in known_solvers
    ), f"Unknown solver: {name}. Supported solvers: {known_solvers}"

    return name
