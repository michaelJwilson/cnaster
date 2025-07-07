import numpy as np
from numba import njit

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

    unique_values, mapping_matrices = [], []

    for s in range(n_spots):
        if total_count.dtype == int:
            pairs = np.unique(np.vstack([obs_count[:, s], total_count[:, s]]).T, axis=0)
        else:
            pairs = np.unique(
                np.vstack([obs_count[:, s], total_count[:, s]]).T.round(decimals=4),
                axis=0,
            )

        unique_values.append(pairs)
        pair_index = {(pairs[i, 0], pairs[i, 1]): i for i in range(pairs.shape[0])}

        # construct mapping matrix
        mat_row = np.arange(n_obs)
        mat_col = np.zeros(n_obs, dtype=int)

        for i in range(n_obs):
            if total_count.dtype == int:
                tmpidx = pair_index[(obs_count[i, s], total_count[i, s])]
            else:
                tmpidx = pair_index[
                    (obs_count[i, s], total_count[i, s].round(decimals=4))
                ]
            mat_col[i] = tmpidx

        mapping_matrices.append(
            scipy.sparse.csr_matrix((np.ones(len(mat_row)), (mat_row, mat_col)))
        )

    return unique_values, mapping_matrices