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

    px = np.linspace(0, 1, 1 + x_part)
    px[-1] += 0.01
    px = px[1:]

    # NB min. to max. x values of all spots (meeting tumor threshold).
    xrange = [np.min(range_coords[:, 0]), np.max(range_coords[:, 0])]

    # NB bin x values into positions and return an appropriate indexing.
    xdigit = np.digitize(
        coords[:, 0], xrange[0] + (xrange[1] - xrange[0]) * px, right=True
    )

    # NB same for y.
    py = np.linspace(0, 1, y_part + 1)
    py[-1] += 0.01
    py = py[1:]

    yrange = [np.min(range_coords[:, 1]), np.max(range_coords[:, 1])]
    ydigit = np.digitize(
        coords[:, 1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True
    )

    initial_clone_index = []
    clone_id = 0

    clone_assignment = np.full(coords.shape[0], -1)

    for xid in range(x_part):
        for yid in range(y_part):
            idx = np.where((xdigit == xid) & (ydigit == yid))[0]

            initial_clone_index.append(idx)
            clone_assignment[idx] = clone_id
            clone_id += 1

    # NB initial clones assigned according to grid partitioning given x_part, y_part; list of lists.
    return initial_clone_index, clone_assignment


def initialize_clones(
    coords, sample_ids, x_part, y_part, single_tumor_prop=None, threshold=None
): 
    logger.info(f"Initializing clones given fixed grid partitions and max. sample_id={np.max(sample_ids)}")

    initial_clone_index = []

    # NB for all slices.
    for s in range(1 + np.max(sample_ids)):
        logger.debug(f"Solving for sample_id={s}")
        
        # NB sample_ids idx for all spots in this slice.
        index = np.where(sample_ids == s)[0]

        if len(index) == 0:
            logger.error(f"Expected at least one spot in slice {s}.")
            raise RuntimeError()

        # NB tumor_proportion for each spot in this slice.
        this_tumor_prop = (
            single_tumor_prop[index] if single_tumor_prop is not None else None
        )

        tmp_clone_index, _ = fixed_rectangle_partition(
            coords[index, :],
            x_part,
            y_part,
            this_tumor_prop,
            threshold=threshold,
        )

        for x in tmp_clone_index:
            initial_clone_index.append(index[x])

    logger.info(
        f"Initialized {len(initial_clone_index)} clones given x_part,y_part={x_part},{y_part}."
    )
    
    return initial_clone_index


def sufficient_umis_initial_clone(
    coords,
    spot_gene_umis,
    sample_list,
    sample_ids,
    min_clone_umis=5_000_000,  # 5_000_000
    acceptance=1.0,
    max_growth_rounds=50,
    random_state=0,
):
    logger.info(
        f"Assigning initial clones based on total spot UMIs, min_clone_umis={min_clone_umis}, acceptance={acceptance} and max_growth_rounds={max_growth_rounds}"
    )

    # TODO HACK
    np.random.seed(random_state)

    # NB across multiple slices.
    n_spots = coords.shape[0]

    # NB -1 means unassigned
    clone_assignment = np.full(n_spots, -1)
    clone_id = 0

    # TODO HACK axis?
    spot_counts = np.sum(spot_gene_umis, axis=0)

    for i, sname in enumerate(sample_list):
        index = np.where(sample_ids == i)[0]
        this_coords = np.array(coords[index, :])
        this_spot_counts = spot_counts[index]

        logger.info(
            f"Solving initial assignment of sample/slice {sname} with {len(this_coords)} spots median spot UMIs {np.median(this_spot_counts)}"
        )

        # NB assignments for this sample/slice.
        assigned = np.zeros(len(index), dtype=bool)

        while not np.all(assigned):
            # NB pick the unassigned spot with the largest UMI count
            unassigned_idx = np.where(~assigned)[0]
            seed_idx = np.random.choice(unassigned_idx)

            group, group_umis = {seed_idx}, this_spot_counts[seed_idx]
            num_rounds = 0

            last_dist = np.inf

            # NB grow group by adding nearest unassigned neighbors until MIN_CLONE_UMIS is reached
            while group_umis < min_clone_umis and len(group) < len(index):
                # NB find unassigned neighbors (by Euclidean distance from seed.)
                dists = np.linalg.norm(
                    this_coords[unassigned_idx] - this_coords[seed_idx], axis=1
                )
                sorted_dists = np.argsort(dists)
                sorted_neighbors = unassigned_idx[sorted_dists]

                for dist, neighbor in zip(sorted_dists, sorted_neighbors):
                    # NB guard against disjoint groups.
                    if dist > 5.0 * last_dist:
                        break

                    if neighbor not in group:
                        group.add(neighbor)
                        group_umis += this_spot_counts[neighbor]

                    if group_umis >= min_clone_umis:
                        break

                    last_dist = dist

                if num_rounds == max_growth_rounds:
                    logger.warning(
                        f"Max growth rounds reached for clone {clone_id} in sample {sname}."
                    )
                    break

                num_rounds += 1

            # NB assign clone_id to these spots
            for g in group:
                assigned[g] = True
                clone_assignment[index[g]] = clone_id

            logger.info(f"Assigned {len(group)} spots to initial clone {clone_id}")

            clone_id += 1

    logger.info(f"Solved for initial clones.")

    initial_clone_index = [np.where(clone_assignment == i)[0] for i in range(clone_id)]

    return initial_clone_index, clone_assignment, spot_counts


# TODO!! spatially contigous clones?
def rectangle_initialize_initial_clone(coords, n_clones, random_state=0):
    # TODO
    np.random.seed(random_state)

    logger.info(
        f"Solving for non-contiguous clone initialization for {n_clones} clones."
    )

    # NB partition x and y range into ~n_clones based on Dirichlet sampling.
    p = int(np.ceil(np.sqrt(n_clones)))

    if n_clones > 1:
        # NB e.g. [0.22, 0.28, 0.25, 0.25], non-negative, sum to unity, Dirichlet sampled.
        px = np.random.dirichlet(np.ones(p) * 10)
        px[-1] += 1e-4

        # NB set xrange as from 5% to 95% percentile of input coords (all slices).
        xrange = [np.percentile(coords[:, 0], 5), np.percentile(coords[:, 0], 95)]

        # NB x positions to dice up input coords.
        xboundary = xrange[0] + (xrange[1] - xrange[0]) * np.cumsum(px)
        xboundary[-1] = np.max(coords[:, 0]) + 1

        # NB x bin for each input (x,y) given x dicing.
        xdigit = np.digitize(coords[:, 0], xboundary, right=True)

        # NB same for y.
        py = np.random.dirichlet(np.ones(p) * 10)
        py[-1] += 1e-4

        yrange = [np.percentile(coords[:, 1], 5), np.percentile(coords[:, 1], 95)]

        yboundary = yrange[0] + (yrange[1] - yrange[0]) * np.cumsum(py)
        yboundary[-1] = np.max(coords[:, 1]) + 1

        ydigit = np.digitize(coords[:, 1], yboundary, right=True)

        # NB partitioned the space into unequal sized blocks.
        block_id = xdigit * p + ydigit
    else:
        block_id = np.zeros(len(coords), dtype=int)
        clone_id = np.zeros(len(coords), dtype=int)

        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]

        logger.info(f"Solved for clone initialization for {n_clones} clones.")

        return initial_clone_index, clone_id

    # NB assigning initial blocks to n_clones (note that if sqrt(n_clone) is not an integer,
    #    multiple blocks can be assigned to a given clone).
    while True:
        # NB assign p^2 initial blocks (randomly) to n_clones.
        block_clone_map = np.random.randint(low=0, high=n_clones, size=p**2)

        # NB its possible a given clone was not assigned ...
        while len(np.unique(block_clone_map)) < n_clones:
            # NB number of blocks assigned to each clone, currently.
            bc = np.bincount(block_clone_map, minlength=n_clones)

            assert np.any(bc == 0)

            # NB take a block from the most-sampled clone and give to an unassigned.
            block_clone_map[np.where(block_clone_map == np.argmax(bc))[0][0]] = (
                np.where(bc == 0)[0][0]
            )

        # NB create a map of block id to clone id.
        block_clone_map = {i: block_clone_map[i] for i in range(len(block_clone_map))}

        # NB maps spots to clones via blocks.
        clone_id = np.array([block_clone_map[i] for i in block_id])

        # NB list of lists: block ids per clone.
        initial_clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]

        # NB min. number of blocks assigned to a given clone is at least 20% of an equal
        #    assignment of spots to clones.
        if (
            np.min([len(x) for x in initial_clone_index])
            > 0.2 * coords.shape[0] / n_clones  # MAGIC.
        ):
            break

    logger.info(f"Solved for clone initialization for {n_clones} clones.")

    return initial_clone_index, clone_id


def compute_adjacency_mat_v2(coords, unit_xsquared=9, unit_ysquared=3, ratio=1):
    """
    Simple distance based adjacency assuming distance scaling factors, unit_xsquared,
    unit_ysquared.
    """
    # NB x,y separations for all spot pairs.
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    # NB arbitrary normalized. y different than x!
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    # NB (# spot, # spot) adjacency matrix.
    A = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.int8)

    logger.info(f"Solving for adjacency matrix with ratio={ratio} and unit_xsquared={unit_xsquared}, unit_ysquared={unit_ysquared}")
    
    # NB loop over spots (across slices).
    for i in range(coords.shape[0]):
        indexes = np.where(
            pairwise_squared_dist[i, :] <= ratio * (unit_xsquared + unit_ysquared)
        )[0]

        # NB drop the spot itself.
        indexes = np.array([j for j in indexes if j != i])

        if len(indexes) > 0:
            A[i, indexes] = 1

    return scipy.sparse.csr_matrix(A)


def compute_weighted_adjacency(
    coords, unit_xsquared=9, unit_ysquared=3, bandwidth=12, decay=5
):
    # NB x,y separations for all spot pairs.
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    # NB arbitrary normalized. y different than x!
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    logger.info(
        f"Solving for inter-slice? Potts adjacency with exponential kernel based on squared distance with bandwidth,decay={bandwidth},{decay}."
    )

    kern = np.exp(-((pairwise_squared_dist / bandwidth) ** decay))

    # NB (spot, spot) adjacency.
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
    logger.info(
        f"Assigning adjaceny matrix based on read counts, assuming unit_xsquared,unit_ysquared={unit_xsquared},{unit_ysquared}."
    )

    # NB x_dist for every spot pair.
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]

    # NB y_dist for every spot pair.
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    # NB x and y dists have independent scale factors.
    tmp_pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    # NB sets the diagonal (self-distances) to the maximum so they are not considered as nearest neighbors.
    # TODO np.inf
    np.fill_diagonal(tmp_pairwise_squared_dist, np.max(tmp_pairwise_squared_dist))

    # NB given the minimum neighbor distance for all spots, find the median and normalize by the sum of scaling factors -
    #    used to set a baseline for neighborhood size.
    base_ratio = np.median(np.min(tmp_pairwise_squared_dist, axis=0)) / (
        unit_xsquared + unit_ysquared
    )

    s_ratio = 0

    for ratio in range(10):
        smooth_mat = compute_adjacency_mat_v2(
            coords, unit_xsquared, unit_ysquared, ratio * base_ratio
        )

        # NB each spot pooled with itself?
        smooth_mat.setdiag(1)

        if np.median(np.sum(smooth_mat > 0, axis=0).A.flatten()) > maxspots_pooling:
            # NB logic is previous once threshold is crossed.
            s_ratio = ratio - 1
            logger.info(
                f"Solved for smooth. mat when spots pooled by distance such that median of pooled spots (per spot) > {maxspots_pooling}."
            )
            break

        s_ratio = ratio

    smooth_mat = compute_adjacency_mat_v2(
        coords, unit_xsquared, unit_ysquared, s_ratio * base_ratio
    )

    smooth_mat.setdiag(1)

    # MAGIC see below.
    logger.info(
        f"Assuming max. length scale (band width) = {15 * (unit_xsquared + unit_ysquared)}"
    )

    for bandwidth in np.arange(
        unit_xsquared + unit_ysquared,  # NB sq. hypotenuse
        15 * (unit_xsquared + unit_ysquared),  # MAGIC
        unit_xsquared + unit_ysquared,
    ):
        # NB distance-based kernel adjacency.
        adjacency_mat = compute_weighted_adjacency(
            coords, unit_xsquared, unit_ysquared, bandwidth=bandwidth
        )

        adjacency_mat.setdiag(1)

        # NB where smooth connection is stronger than exponential, we rely on smooth.
        adjacency_mat = adjacency_mat - smooth_mat
        adjacency_mat[adjacency_mat < 0] = 0

        if np.median(np.sum(adjacency_mat, axis=0).A.flatten()) >= 6:  # MAGIC
            logger.info(
                f"Solved for adjacency matrix with length scale {bandwidth} and median of total edge > 6 (MAGIC)."
            )
            break

    return smooth_mat, adjacency_mat


def renormalize_adjacency_mat(adjacency_mat):
    num_edges, total_edge_weight = [], []

    for row in list(adjacency_mat.tolil()):
        num_edges.append(row.nnz)
        total_edge_weight.append(row.sum())

    num_edges = np.array(num_edges)
    total_edge_weight = np.array(total_edge_weight)

    us, cnts = np.unique(num_edges, return_counts=True)
    med_num_edges = np.median(num_edges)

    logger.info(
        f"Found node degree distribution with median {med_num_edges}:\n{us}\n{cnts}"
    )

    us, cnts = np.unique(total_edge_weight, return_counts=True)
    med_edge_weight = np.median(total_edge_weight)

    logger.info(
        f"Found edge weight distribution with median {med_edge_weight}:\n{us}\n{cnts}"
    )

    adj = adjacency_mat.tocsr().astype(np.float64)
    row_sums = np.asarray(adj.sum(axis=1)).ravel()  # shape (n_rows,)

    indptr = adj.indptr
    data = adj.data
    n_rows = adj.shape[0]

    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]

        if start == end:
            # empty row
            continue

        rs = row_sums[i]

        if rs == 0.0:
            # nothing to scale
            continue

        scale = med_edge_weight / rs
        data[start:end] *= scale

    adj.eliminate_zeros()

    logger.info(f"Normalized adjacency_mat:\n{adj}")

    return adj


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
    unit_xsquared=9,
    unit_ysquared=3,
):
    logger.info("Solving for multi-slice adjacency (and smooth) matrix.")

    # NB smooth_mat contains the edges of spots that are directly pooled.
    adjacency_mat, smooth_mat = [], []

    # NB loop over slices.
    for i, sname in enumerate(sample_list):
        # NB spots per slice.
        index = np.where(sample_ids == i)[0]

        # NB (x,y) for these spots.
        this_coords = np.array(coords[index, :])

        tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts(
            this_coords,
            single_total_bb_RD[:, index],
            maxspots_pooling=maxspots_pooling,
            unit_xsquared=unit_xsquared,
            unit_ysquared=unit_ysquared,
        )

        adjacency_mat.append(tmpadjacency_mat.toarray())
        smooth_mat.append(tmpsmooth_mat.toarray())

    # NB sets block diagonals corresponding to inter-slice.
    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)

    # NB realize as sparse.
    adjacency_mat = scipy.sparse.csr_matrix(adjacency_mat)

    # NB add intra-slice adjacency.
    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    # NB realize as block diagonal for inter-slice pooling.
    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix(smooth_mat)

    return adjacency_mat, smooth_mat
