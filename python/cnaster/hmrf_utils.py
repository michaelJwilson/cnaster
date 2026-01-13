import numpy as np
import logging
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

class Assignments:
    def __init__(self, assignment):
        self.assignment = None
        self.update(assignment)

    def update(self, assignment):
        assignment = np.asarray(assignment)
        unique = np.unique(assignment)
        expected = np.arange(unique.size)
        assert np.array_equal(unique, expected), (
            f"Assignment must be monotonically increasing from 0 with no gaps. "
            f"Found unique={unique}, expected={expected}"
        )
        self.assignment = assignment

    def get(self):
        return self.assignment

    def __len__(self):
        return len(self.assignment)

    def __getitem__(self, idx):
        return self.assignment[idx]


# TODO validate
def cast_csr(csr_matrix):
    result = []

    for i in range(csr_matrix.shape[0]):
        start_idx = csr_matrix.indptr[i]
        end_idx = csr_matrix.indptr[i + 1]

        row_data = []

        for idx in range(start_idx, end_idx):
            col = csr_matrix.indices[idx]
            val = csr_matrix.data[idx]
            row_data.append((col, val))

        result.append(row_data)

    return result


def clone_stack_obs(
    X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
):
    # NB vertical stacking of X, base_nb_mean, total_bb_RD, tumor_prop across clones,
    # i.e. reshape observation data from (n_obs, 2, n_clones) to (n_obs * n_clones, 2, 1)
    clone_stack_X = np.vstack(
        [X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]
    ).T.reshape(-1, 2, 1)

    # NB vertical stacking by clone, cast to column.
    clone_stack_base_nb_mean = base_nb_mean.flatten("F").reshape(-1, 1)
    clone_stack_total_bb_RD = total_bb_RD.flatten("F").reshape(-1, 1)

    # NB replicate lengths N clone times, as derived from X - clone num. may change.
    clone_stack_lengths = np.tile(lengths, X.shape[2])
    clone_stack_sitewise_transmat = np.tile(log_sitewise_transmat, X.shape[2])

    # NB per-clone tumor prop. repeated num_obs times.
    stack_tumor_prop = (
        np.repeat(tumor_prop, X.shape[0]).reshape(-1, 1)
        if tumor_prop is not None
        else None
    )

    logger.info(f"Stacked X from shape {X.shape} to {clone_stack_X.shape}.")
    logger.info(
        f"Stacked total_bb_RD from shape {total_bb_RD.shape} to {clone_stack_total_bb_RD.shape}."
    )

    return (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        clone_stack_lengths,
        clone_stack_sitewise_transmat,
        stack_tumor_prop,
    )


def get_clone_indices(assignments, clone_ids):
    """
    Return a list of arrays, each containing the indices of spots assigned to each clone ID.

    Args:
        assignments (np.ndarray): Array of clone assignments for each spot.
        clone_ids (array-like): Iterable of clone IDs to extract indices for.

    Returns:
        List[np.ndarray]: List of index arrays, one per clone ID.
    """
    return [np.where(assignments == cid)[0] for cid in clone_ids]


def get_clone_assignment(coords, clone_indices):
    n_spots = sum(len(indices) for indices in clone_indices)

    assert n_spots == len(
        coords
    ), "Total number of spots does not match length of coords."

    assignment = np.full(len(coords), -1, dtype=int)

    for idx, indices in enumerate(clone_indices):
        assignment[indices] = idx

    return assignment
