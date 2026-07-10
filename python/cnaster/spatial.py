from collections import namedtuple

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from cnaster.annotation import get_clone_label_annotation
from cnaster.config import start_time
from cnaster.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


# TODO respect alignment.
def rectangle_partition(
    coords, x_part, y_part, single_tumor_prop=None, threshold=0.5, random_state=None
):
    if single_tumor_prop is not None:
        idx_tumor = np.where(single_tumor_prop >= threshold)[0]
        range_coords = coords[idx_tumor]
    else:
        range_coords = coords

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        px = np.sort(rng.uniform(0, 1, x_part))
        px[-1] = 1.01
    else:
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
    if random_state is not None:
        py = np.sort(rng.uniform(0, 1, y_part))
        py[-1] = 1.01
    else:
        py = np.linspace(0, 1, y_part + 1)
        py[-1] += 0.01
        py = py[1:]

    yrange = [np.min(range_coords[:, 1]), np.max(range_coords[:, 1])]
    ydigit = np.digitize(
        coords[:, 1], yrange[0] + (yrange[1] - yrange[0]) * py, right=True
    )

    # logger.info(f"Solved for xrange={xrange}, yrange={yrange}")
    # logger.info(
    #     f"Solved for x-partitions={xrange[0] + (xrange[1] - xrange[0]) * px}, y partitions={yrange[0] + (yrange[1] - yrange[0]) * py}"
    # )

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


def best_equal_partition(
    coords, x_part, y_part, single_tumor_prop=None, threshold=0.5, n_trials=10_000
):
    best_var = np.inf
    best_index = None
    best_assignment = None

    for trial in range(n_trials):
        initial_clone_index, clone_assignment = rectangle_partition(
            coords,
            x_part,
            y_part,
            single_tumor_prop=single_tumor_prop,
            threshold=threshold,
            random_state=trial,
        )
        sizes = [len(idx) for idx in initial_clone_index]
        var = np.var(sizes)
        if var < best_var:
            best_var = var
            best_index = initial_clone_index
            best_assignment = clone_assignment

            logger.info(
                f"New best partition at trial {trial}: variance={best_var:.2f}, sizes={sizes}"
            )

    logger.info(f"Best partition variance after {n_trials} trials: {best_var:.2f}")
    return best_index, best_assignment


def initialize_clones(
    coords,
    sample_ids,
    x_part,
    y_part,
    single_tumor_prop=None,
    threshold=None,
    random_state=None,
    config=None,
):
    if config is not None:
        # NB assumes the known clone labels for phasing.
        if config.annotation.clone_label is not None:
            clone_annotation, _ = get_clone_label_annotation(config)

            # NB assumes the known clone labels for phasing.
            return clone_annotation

    logger.info(
        f"Initializing clones given fixed grid partitions and max. sample_id={np.max(sample_ids)}"
    )

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

        tmp_clone_index, _ = rectangle_partition(
            coords[index, :],
            x_part,
            y_part,
            this_tumor_prop,
            threshold=threshold,
            random_state=random_state,
        )

        for x in tmp_clone_index:
            initial_clone_index.append(index[x])

    logger.info(
        f"Initialized {len(initial_clone_index)} clones given x_part,y_part={x_part},{y_part}."
    )

    return initial_clone_index


# TODO!! spatially contigous clones?
def initialize_rectangular_clones(coords, n_clones, random_state=0):
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


def construct_lattice_adjacency(
    coords,
    maxspots_pooling=7,
    unit_xsquared=9,
    unit_ysquared=3,
    coordination_num=8,
):
    # NB called per slice.
    # coordination_num = summarize_lattice_structure(
    #     coords, sample_ids=np.zeros(len(coords)), sample_list=[None]
    # )

    # if coordination_num < min_coordination_num:
    #     logger.warning(f"Assuming minimum coordination number={min_coordination_num}.")
    # 
    #     coordination_num = min_coordination_num

    logger.info(
        f"Assigning lattice adjacency matrix with coordination_num={coordination_num}, "
        f"assuming unit_xsquared,unit_ysquared={unit_xsquared},{unit_ysquared}."
    )

    n_spots = coords.shape[0]

    scaled_coords = coords.copy().astype(float)
    scaled_coords[:, 0] *= np.sqrt(unit_xsquared)
    scaled_coords[:, 1] *= np.sqrt(unit_ysquared)

    logger.info(f"Building KD-tree for efficient nearest neighbor search")

    tree = cKDTree(scaled_coords)

    # NB query (k+1) nearest neighbors as includes self.
    _, indices = tree.query(scaled_coords, k=coordination_num + 1)

    indices = indices[:, 1:]

    logger.info(f"Constructed nearest neighbor indices via KD-tree")

    # NB smooth matrix: identity (each spot pools only itself)
    smooth_mat = scipy.sparse.identity(n_spots, dtype=np.int8, format="csr")

    logger.warning(f"Assumed identity smooth mat.")

    logger.info(f"Constructing adjacency matrix.")

    # nearest_indices = np.argpartition(pairwise_squared_dist, coordination_num, axis=1)[:, :coordination_num]

    rows = np.repeat(np.arange(n_spots), coordination_num)
    cols = indices.flatten()
    data = np.ones(len(rows), dtype=np.float64)

    adjacency_mat = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))

    # TODO
    # num_neighbors = np.sum(adjacency_mat > 0, axis=1).A.flatten()

    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getnnz.html
    num_neighbors = adjacency_mat.getnnz(axis=1)

    # NB lattice adjacency: min=2, median=2.0, max=4 neighbors per spot.
    logger.info(
        f"Lattice adjacency: min={np.min(num_neighbors)}, "
        f"median={np.median(num_neighbors):.1f}, "
        f"max={np.max(num_neighbors)} neighbors per spot"
    )

    return smooth_mat, adjacency_mat


# @cacher("adjacency.hdf5")
def construct_multislice_lattice_adjacency(
    sample_ids,
    sample_list,
    coords,
    # exp_counts,
    across_slice_adjacency_mat,
    # construct_adjacency_method,
    maxspots_pooling,
    # construct_adjacency_w,
    unit_xsquared=9,
    unit_ysquared=3,
):
    logger.info("Solving for multi-slice adjacency (and spot-pooling) matrix.")

    # NB smooth_mat contains the edges of spots that are directly pooled.
    adjacency_mat, smooth_mat = [], []

    for i, _ in enumerate(sample_list):
        # NB spots per slice.
        index = np.where(sample_ids == i)[0]

        # NB (x,y) for these spots.
        this_coords = np.array(coords[index, :])

        # NB smooth and adjacency matrices for this slice.
        tmpsmooth_mat, tmpadjacency_mat = construct_lattice_adjacency(
            this_coords,
            maxspots_pooling=maxspots_pooling,
            unit_xsquared=unit_xsquared,
            unit_ysquared=unit_ysquared,
        )

        adjacency_mat.append(tmpadjacency_mat.toarray())
        smooth_mat.append(tmpsmooth_mat.toarray())

    # NB realize as block diagonal for inter-slice pooling.
    smooth_mat = scipy.linalg.block_diag(*smooth_mat)
    smooth_mat = scipy.sparse.csr_matrix(smooth_mat)

    # NB sets block diagonals corresponding to inter-slice.
    adjacency_mat = scipy.linalg.block_diag(*adjacency_mat)
    adjacency_mat = scipy.sparse.csr_matrix(adjacency_mat)

    # NB add intra-slice adjacency.
    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    logger.info("Solving for multi-slice adjacency (and spot-pooling) matrix.")

    Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])

    return Adjacency(adjacency_mat=adjacency_mat, smooth_mat=smooth_mat)


def initialize_rdr_clone_refininement(
    merged_baf_assignment, coords, single_total_bb_RD, n_obs, config
):
    n_spots = len(merged_baf_assignment)
    n_baf_clones = len(np.unique(merged_baf_assignment))

    # TODO HACK assert (0, ..., N-1) for clone labels.

    splits_per_baf, total_rdr_clones = [], 0

    for bafc in range(n_baf_clones):
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        sufficient_snp_umi = np.sum(single_total_bb_RD[:, idx_spots]) >= 20 * n_obs

        n_splits = config.hmrf.n_clones_rdr if sufficient_snp_umi else 1
        splits_per_baf.append(n_splits)
        total_rdr_clones += n_splits

    logger.info(
        f"Global hmrf will optimize {total_rdr_clones} rdr-refined clones given {n_baf_clones} baf-identified clones."
    )

    global_initial_assignment = np.zeros(n_spots, dtype=np.int32)

    # NB one-hot allowed (rdr-refined) clones.
    allowed_clones = np.zeros((n_spots, total_rdr_clones), dtype=bool)
    global_clone_offset = 0

    for bafc, n_splits in enumerate(splits_per_baf):
        idx_spots = np.where(merged_baf_assignment == bafc)[0]

        # TODO HACK define initializer; define seed.
        initial_clone_index, _ = initialize_rectangular_clones(
            coords[idx_spots],
            n_splits,
            random_state=config.hmm.gmm_random_state,
        )

        for local_c, local_idx in enumerate(initial_clone_index):
            global_c = global_clone_offset + local_c

            global_spot_ids = idx_spots[local_idx]
            global_initial_assignment[global_spot_ids] = global_c

        allowed_clones[
            idx_spots, global_clone_offset : global_clone_offset + n_splits
        ] = True
        global_clone_offset += n_splits

    return global_initial_assignment, allowed_clones, total_rdr_clones


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
            continue

        scale = med_edge_weight / rs
        data[start:end] *= scale

    adj.eliminate_zeros()

    logger.info(f"Normalized adjacency_mat:\n{adj}")

    return adj