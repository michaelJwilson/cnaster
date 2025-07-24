import logging

import numpy as np
import scipy
from numba import njit
from cnaster.config import get_global_config

logger = logging.getLogger(__name__)


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


def convert_params(mean, std):
    p = mean / std**2
    n = mean * p / (1.0 - p)

    return n, p


def calc_sparsity(csr_matrix):
    total_elements = csr_matrix.shape[0] * csr_matrix.shape[1]
    non_zero_elements = csr_matrix.size

    return (total_elements - non_zero_elements) / total_elements


def construct_unique_matrix(obs_count, total_count):
    """
    Attributes
    ----------
    allele_count : array, shape (n_observations, n_spots)
        Observed A allele counts per SNP per spot.

    total_bb_RD : array, shape (n_observations, n_spots)
        Total SNP-covering reads per SNP per spot.
    """
    n_obs = obs_count.shape[0]
    n_spots = obs_count.shape[1]
    decimals = name = get_global_config().hmm.compression_decimals

    unique_values, mapping_matrices = [], []

    mean_validity, mean_compression, mean_sparsity = 0.0, 0.0, 0.0

    for s in range(n_spots):
        valid = total_count[:, s] > 0
        counts = np.vstack([obs_count[:, s], total_count[:, s]]).T

        mean_validity += np.mean(valid)
        
        # TODO BUG fails for numpy cases; not np.issubdtype(total_count.dtype, np.integer)
        if total_count.dtype != int:
            counts = counts.round(decimals=decimals)

        pairs = np.unique(counts, axis=0)

        logger.info(f"Found {len(pairs)} unique pairs with {100. * np.mean(valid)}% valid:\n{pairs}")
        
        mean_compression += (1. - len(pairs) / n_obs)

        unique_values.append(pairs)
        pair_index = {(pairs[i, 0], pairs[i, 1]): i for i in range(pairs.shape[0])}

        # NB construct mapping matrix with shape (n_obs, n_unique_pairs);
        #    one-hot of obs. to compressed.
        mat_row = np.arange(n_obs)
        mat_col = np.zeros(n_obs, dtype=int)

        for i in range(n_obs):
            if total_count.dtype == int:
                tmpidx = pair_index[(obs_count[i, s], total_count[i, s])]
            else:
                tmpidx = pair_index[
                    (obs_count[i, s], total_count[i, s].round(decimals=decimals))
                ]
            mat_col[i] = tmpidx

        # NB num. columns set by max(mat_col).
        csr_matrix = scipy.sparse.csr_matrix(
            (np.ones(len(mat_row)), (mat_row, mat_col))
        )

        mean_sparsity += calc_sparsity(csr_matrix)
        
        # Example usage:
        #   e.g.  convert posteriors from observation space to the compressed space
        # .        tmp = (scipy.sparse.csr_matrix(gamma) @ mapping_matrices[s]).toarray()
        mapping_matrices.append(csr_matrix)

    mean_validity /= n_spots
    mean_compression /= n_spots
    mean_sparsity /= n_spots

    msg = f"Constructed unique count compression with mean validity: {100. * mean_validity}, mean compression rate (decimals={decimals}): {100. * mean_compression:.4f}%, "
    msg += f"as represented by {n_spots} sparse matrices with mean sparsity {100. * mean_sparsity:.2f}%."
    
    logger.info(msg)

    return unique_values, mapping_matrices


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

    log_gamma = np.zeros((n_states, n_obs))
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
    known_solvers = ("newton", "bfgs", "lbfgs", "powell", "nm", "cg", "ncg")

    name = get_global_config().hmm.solver

    assert (
        name in known_solvers
    ), f"Unknown solver: {name}. Supported solvers: {known_solvers}"

    return name
