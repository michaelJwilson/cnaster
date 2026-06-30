import numpy as np
import time
import csv
from pathlib import Path

from dataclasses import dataclass, asdict, field
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# class ValidAssignment:
#     def __init__(self, assignment):
#         self.assignment = None
#         self.set(assignment)

#     def get(self):
#         return self.assignment

#     def set(self, assignment):
#         assignment = np.asarray(assignment)
#         unique = np.unique(assignment)
#         expected = np.arange(unique.size)
#         assert np.array_equal(unique, expected), (
#             f"Assignment must be monotonically increasing from 0 with no gaps. "
#             f"Found unique={unique}, expected={expected}"
#         )
#         self.assignment = assignment

#     def __len__(self):
#         return len(self.assignment)

#     def __getitem__(self, idx):
#         return self.assignment[idx]


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

'''
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
'''

def clone_stack_obs(
    X, base_nb_mean, total_bb_RD, lengths, log_sitewise_transmat, tumor_prop
):
    n_obs, n_comp, n_clones = X.shape

    # NB. transpose moves clones to the first dimension: (n_clones, n_obs, 2)
    #     reshaping to (-1, 2, 1) perfectly mimics the flatten("F") + vstack logic 
    #     but strictly requires only one C-level memory copy.
    clone_stack_X = X.transpose(2, 0, 1).reshape(-1, n_comp, 1)

    # NB transposing (n_obs, n_clones) to (n_clones, n_obs) before reshaping
    #    cleanly achieves the exact same result as flatten("F").
    clone_stack_base_nb_mean = base_nb_mean.T.reshape(-1, 1)
    clone_stack_total_bb_RD = total_bb_RD.T.reshape(-1, 1)

    # NB replicate lengths and transmats n_clones times (e.g., [A, B] -> [A, B, A, B])
    clone_stack_lengths = np.tile(lengths, n_clones)
    clone_stack_sitewise_transmat = np.tile(log_sitewise_transmat, n_clones)

    # NB repeat scalar per clone n_obs times (e.g., [A, B] -> [A, A, B, B])
    if tumor_prop is not None:
        stack_tumor_prop = np.repeat(tumor_prop, n_obs).reshape(-1, 1)
    else:
        stack_tumor_prop = None

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


@dataclass
class hmrf_perf_entry:
    optimizer: str
    cost: float
    best_cost: float
    iteration: int = 0
    temp: float = np.nan
    acceptance: float = 1.0
    ncluster: int = 1
    nedit: int = 0
    clone_split: np.ndarray = field(default_factory=lambda: np.array([-1]))

    def as_dict(self):
        d = asdict(self)
        d["optimizer"] = d["optimizer"].ljust(15)
        d["cost"] = "{:+.6e}".format(self.cost)
        d["best_cost"] = "{:+.6e}".format(self.best_cost)
        d["iteration"] = str(self.iteration)
        d["temp"] = (
            "Inf".ljust(10) if np.isinf(self.temp) else "{:.4e}".format(self.temp)
        )
        d["acceptance"] = "{:.4e}".format(self.acceptance)
        d["ncluster"] = str(self.ncluster)
        d["nedit"] = "{:d}".format(self.nedit)
        d["clone_split"] = ",".join("{:.8f}".format(x) for x in self.clone_split)

        return d

    def log(self, filename="cnaster_hmrf.perf"):
        perf_dict = self.as_dict()
        perf_file = Path(filename)

        perf_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fieldnames = ["timestamp"] + [k for k in perf_dict.keys() if k != "timestamp"]

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

            if not perf_file.exists():
                writer.writeheader()

            writer.writerow(perf_dict)
