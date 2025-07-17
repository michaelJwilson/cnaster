import copy
import logging

import numpy as np
import scipy.linalg
import scipy.sparse

logger = logging.getLogger(__name__)


# TODO respect alignment.
def fixed_rectangle_partition(
    coords, x_part, y_part, single_tumor_prop=None, threshold=0.5
):
    """
    Initialize rectangular grid partitioning of coordinates.

    Parameters:
    -----------
    coords : array-like, shape (n, 2)
        Coordinate array with x, y positions
    x_part : int
        Number of partitions in x direction
    y_part : int
        Number of partitions in y direction
    single_tumor_prop : array-like, optional
        Tumor proportion values. If provided, only coordinates with
        tumor_prop > threshold are used to determine the coordinate ranges
    threshold : float, default=0.5
        Threshold for tumor proportion filtering

    Returns:
    --------
    initial_clone_index : list
        List of arrays containing indices for each grid cell
    """
    if single_tumor_prop is not None:
        idx_tumor = np.where(single_tumor_prop >= threshold)[0]
        range_coords = coords[idx_tumor]
    else:
        range_coords = coords

    px = np.linspace(0, 1, x_part + 1)
    px[-1] += 0.01
    px = px[1:]

    xrange = [np.min(range_coords[:, 0]), np.max(range_coords[:, 0])]
    xdigit = np.digitize(
        coords[:, 0], xrange[0] + (xrange[1] - xrange[0]) * px, right=True
    )

    py = np.linspace(0, 1, y_part + 1)
    py[-1] += 0.01
    py = py[1:]

    yrange = [np.min(range_coords[:, 1]), np.max(range_coords[:, 1])]
    ydigit = np.digitize(
        coords[:, 1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True
    )

    initial_clone_index = []

    for xid in range(x_part):
        for yid in range(y_part):
            initial_clone_index.append(np.where((xdigit == xid) & (ydigit == yid))[0])

    return initial_clone_index


def initialize_clones(
    coords, sample_ids, x_part, y_part, single_tumor_prop=None, threshold=None
):
    initial_clone_index = []

    for s in range(1 + np.max(sample_ids)):
        index = np.where(sample_ids == s)[0]

        if len(index) == 0:
            logger.error(f"Invalid sample_ids found: {sample_ids}")
            raise RuntimeError()

        # NB would be per cell tumor props stacked across samples.
        this_tumor_prop = (
            single_tumor_prop[index] if single_tumor_prop is not None else None
        )

        tmp_clone_index = fixed_rectangle_partition(
            coords[index, :],
            x_part,
            y_part,
            this_tumor_prop,
            threshold=threshold,
        )

        for x in tmp_clone_index:
            initial_clone_index.append(index[x])

    return initial_clone_index


def rectangle_initialize_initial_clone(coords, n_clones, random_state=0):
    np.random.seed(random_state)

    # NB partition x and y range into ~n_clones based on Dirichlet sampling.
    p = int(np.ceil(np.sqrt(n_clones)))

    px = np.random.dirichlet(np.ones(p) * 10)
    px[-1] += 1e-4

    xrange = [np.percentile(coords[:, 0], 5), np.percentile(coords[:, 0], 95)]

    xboundary = xrange[0] + (xrange[1] - xrange[0]) * np.cumsum(px)
    xboundary[-1] = np.max(coords[:, 0]) + 1

    xdigit = np.digitize(coords[:, 0], xboundary, right=True)

    py = np.random.dirichlet(np.ones(p) * 10)
    py[-1] += 1e-4

    yrange = [np.percentile(coords[:, 1], 5), np.percentile(coords[:, 1], 95)]

    yboundary = yrange[0] + (yrange[1] - yrange[0]) * np.cumsum(py)
    yboundary[-1] = np.max(coords[:, 1]) + 1

    ydigit = np.digitize(coords[:, 1], yboundary, right=True)

    # NB partitioned the space into "blocks".
    block_id = xdigit * p + ydigit

    # TODO? assigning blocks to clone (note that if sqrt(n_clone) is not an integer,
    # multiple blocks can be assigned to one clone)
    while True:
        # NB assign blocks to clones.
        block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)

        while len(np.unique(block_clone_map)) < n_clones:
            bc = np.bincount(block_clone_map, minlength=n_clones)

            assert np.any(bc == 0)

            # NB take a block from the over-represented clone and give to the unassigned
            #    clone.
            block_clone_map[np.where(block_clone_map == np.argmax(bc))[0][0]] = (
                np.where(bc == 0)[0][0]
            )

        block_clone_map = {i: block_clone_map[i] for i in range(len(block_clone_map))}
        clone_id = np.array([block_clone_map[i] for i in block_id])
        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]

        if (
            np.min([len(x) for x in initial_clone_index])
            > 0.2 * coords.shape[0] / n_clones
        ):
            break

    return initial_clone_index


def compute_adjacency_mat_v2(coords, unit_xsquared=9, unit_ysquared=3, ratio=1):
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    A = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.int8)

    for i in range(coords.shape[0]):
        indexes = np.where(
            pairwise_squared_dist[i, :] <= ratio * (unit_xsquared + unit_ysquared)
        )[0]

        indexes = np.array([j for j in indexes if j != i])

        if len(indexes) > 0:
            A[i, indexes] = 1

    return scipy.sparse.csr_matrix(A)


def compute_weighted_adjacency(
    coords, unit_xsquared=9, unit_ysquared=3, bandwidth=12, decay=5
):
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    kern = np.exp(-((pairwise_squared_dist / bandwidth) ** decay))

    A = np.zeros((coords.shape[0], coords.shape[0]))

    for i in range(coords.shape[0]):
        indexes = np.where(kern[i, :] > 1e-4)[0]
        indexes = np.array([j for j in indexes if j != i])

        if len(indexes) > 0:
            A[i, indexes] = kern[i, indexes]

    return scipy.sparse.csr_matrix(A)


def choose_adjacency_by_readcounts(
    coords, single_total_bb_RD, maxspots_pooling=7, unit_xsquared=9, unit_ysquared=3
):
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    tmp_pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    np.fill_diagonal(tmp_pairwise_squared_dist, np.max(tmp_pairwise_squared_dist))

    base_ratio = np.median(np.min(tmp_pairwise_squared_dist, axis=0)) / (
        unit_xsquared + unit_ysquared
    )

    s_ratio = 0

    for ratio in range(0, 10):
        smooth_mat = compute_adjacency_mat_v2(
            coords, unit_xsquared, unit_ysquared, ratio * base_ratio
        )

        smooth_mat.setdiag(1)

        if np.median(np.sum(smooth_mat > 0, axis=0).A.flatten()) > maxspots_pooling:
            s_ratio = ratio - 1
            break

        s_ratio = ratio

    smooth_mat = compute_adjacency_mat_v2(
        coords, unit_xsquared, unit_ysquared, s_ratio * base_ratio
    )

    smooth_mat.setdiag(1)

    for bandwidth in np.arange(
        unit_xsquared + unit_ysquared,
        15 * (unit_xsquared + unit_ysquared),
        unit_xsquared + unit_ysquared,
    ):
        adjacency_mat = compute_weighted_adjacency(
            coords, unit_xsquared, unit_ysquared, bandwidth=bandwidth
        )

        adjacency_mat.setdiag(1)

        adjacency_mat = adjacency_mat - smooth_mat
        adjacency_mat[adjacency_mat < 0] = 0

        if np.median(np.sum(adjacency_mat, axis=0).A.flatten()) >= 6:
            logger.info(f"bandwidth: {bandwidth}")
            break

    return smooth_mat, adjacency_mat


def multislice_adjacency(
    sample_ids,
    sample_list,
    coords,
    single_total_bb_RD,
    exp_counts,
    across_slice_adjacency_mat,
    construct_adjacency_method,
    maxspots_pooling,
    construct_adjacency_w,
):
    adjacency_mat, smooth_mat = [], []

    for i, sname in enumerate(sample_list):
        index = np.where(sample_ids == i)[0]
        this_coords = np.array(coords[index, :])

        tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts(
            this_coords,
            single_total_bb_RD[:, index],
            maxspots_pooling=maxspots_pooling,
        )

        adjacency_mat.append(tmpadjacency_mat.toarray())
        smooth_mat.append(tmpsmooth_mat.toarray())

    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)
    adjacency_mat = scipy.sparse.csr_matrix(adjacency_mat)

    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix(smooth_mat)

    return adjacency_mat, smooth_mat
