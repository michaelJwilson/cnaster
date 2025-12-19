import numpy as np
import logging

logger = logging.getLogger(__name__)

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
