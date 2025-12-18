import logging

import numpy as np
import scipy.linalg
import scipy.sparse
from collections import namedtuple
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from cnaster.utils import cacher

logger = logging.getLogger(__name__)


# TODO respect alignment.
def fixed_rectangle_partition(
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

    logger.info(f"Solved for xrange={xrange}, yrange={yrange}")
    logger.info(
        f"Solved for x-partitions={xrange[0] + (xrange[1] - xrange[0]) * px}, y partitions={yrange[0] + (yrange[1] - yrange[0]) * py}"
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
    coords, sample_ids, x_part, y_part, single_tumor_prop=None, threshold=None, random_state=None
):
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

        tmp_clone_index, _ = fixed_rectangle_partition(
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


def summarize_lattice_structure(coords, sample_ids=None, sample_list=None):
    """
    Primitive lattice vectors on a hexagonal lattice are equal length, at angle of 120 deg.
    """
    coordination_number = None

    for i, _ in enumerate(sample_list):
        index = np.where(sample_ids == i)[0]

        this_coords = np.array(coords[index, :]).copy()

        nx = len(np.unique(this_coords[:, 0]))
        ny = len(np.unique(this_coords[:, 1]))

        center_x = np.median(this_coords[:, 0])
        center_y = np.median(this_coords[:, 1])

        # NB median may not be a realized coordinate - find closest.
        center_x_idx = np.argmin(np.abs(this_coords[:, 0] - center_x))
        center_y_idx = np.argmin(np.abs(this_coords[:, 1] - center_y))

        center_x = this_coords[center_x_idx, 0]
        center_y = this_coords[center_y_idx, 1]

        center_xy_idx = np.argmin(
            (this_coords[:, 0] - center_x) ** 2.0
            + (this_coords[:, 1] - center_y) ** 2.0
        )

        center_x = this_coords[center_xy_idx, 0]
        center_y = this_coords[center_xy_idx, 1]

        center_x_dist = this_coords[center_xy_idx, 0] - this_coords[:, 0]
        center_y_dist = this_coords[center_xy_idx, 1] - this_coords[:, 1]

        # NB distance to center - picked as oracle spot to determine nearest neighbor structure.
        center_pairwise_dist = np.sqrt(center_x_dist**2 + center_y_dist**2)

        # NB first is self.
        sorted_indices = np.argsort(center_pairwise_dist)[1:]

        # NB CHECK Bravais lattices in 2d have 2 primitive lattice vectors on a plane.
        sorted_dists = center_pairwise_dist[sorted_indices]
        sorted_neighbors = this_coords[sorted_indices, :]

        unique_dists, unique_cnts = np.unique(sorted_dists, return_counts=True)

        # NB cell positions of Visium HD will not be regular.
        logger.info(f"Found lattice distances from center:\n{unique_dists}\nwith counts:\n{unique_cnts}")

        assert np.all(unique_dists > 0.0)

        # NB expect (2,0) and (1,1) with length 2 and sqrt(2) at angle 45 deg. for hexagonal lattice.
        unique_dists = unique_dists[:2]
        unique_cnts = unique_cnts[:2]

        # TODO hexagonal gives eight as straight up two rows is 2 away, same as left/right.
        if coordination_number is None:
            coordination_number = sum(unique_cnts)
        else:
            assert coordination_number == sum(
                unique_cnts
            ), "Found inconsistent coordination number across samples @ {i}."

        sorted_neighbors = sorted_neighbors[:coordination_number]
        sorted_displacements = sorted_neighbors - np.array([[center_x, center_y]])

        logger.info(f"Sample {i}: estimated lattice spacing nx, ny = {nx:_}, {ny:_}")
        logger.info(
            f"Lattice coordination number={coordination_number} with displacements from center spot at ({center_x:.1f}, {center_y:.1f})=\n{sorted_displacements}"
        )

    return coordination_number


# TODO snp umi requirement
def sufficient_umis_initial_clone(
    coords,
    spot_gene_umis,
    sample_list,
    sample_ids,
    min_clone_umis=500_000,
    max_growth_rounds=50,
    random_state=0,
    prior_clone_assignment=None,
):
    logger.info(
        f"Assigning initial clones based on total spot UMIs, min_clone_umis={min_clone_umis:_} and max_growth_rounds={max_growth_rounds}"
    )

    if prior_clone_assignment is not None:
        raise NotImplementedError()

    summarize_lattice_structure(coords, sample_ids=sample_ids, sample_list=sample_list)

    rand_rng = np.random.default_rng(random_state)

    # NB across multiple slices.
    n_spots = coords.shape[0]

    # NB -1 means unassigned
    clone_assignment = np.full(n_spots, -1)
    clone_id = 0

    # TODO note why transposed?
    spot_counts = np.sum(spot_gene_umis, axis=0)

    for i, _ in enumerate(sample_list):
        index = np.where(sample_ids == i)[0]
        num_spots_slice = len(index)

        this_coords = np.array(coords[index, :])
        this_spot_counts = spot_counts[index]

        logger.info(
            f"Solving initial assignment of sample/slice {i} with {len(this_coords):_} spots median spot UMIs {np.median(this_spot_counts)}"
        )

        # NB assignments for this sample/slice.
        assigned = np.zeros(len(index), dtype=bool)

        while not np.all(assigned):
            num_rounds = 0
            unassigned_idx = np.where(~assigned)[0]

            # NB seed spot to grow a new initial clone.
            seed_idx = rand_rng.choice(unassigned_idx)
            group, group_umis = {seed_idx}, this_spot_counts[seed_idx]

            last_dist = np.inf

            # NB compute distances from seed to all spots on this slice.
            seed_dists = np.linalg.norm(this_coords - this_coords[seed_idx], axis=1)

            initial_group_umis = this_spot_counts[seed_idx].copy()

            # NB grow group by adding nearest unassigned neighbors until MIN_CLONE_UMIS is reached
            while group_umis < min_clone_umis:
                unassigned_idx = np.where(~assigned)[0]

                if len(unassigned_idx) == 0:
                    logger.warning("Assigned all spots on slice.")
                    break

                unassigned_seed_dists = seed_dists[unassigned_idx]

                # NB sort unassigned spots by distance from seed.
                sorted_indices = np.argsort(unassigned_seed_dists)
                sorted_dists = unassigned_seed_dists[sorted_indices]
                sorted_neighbors = unassigned_idx[sorted_indices]

                for _, neighbor in zip(sorted_dists, sorted_neighbors):
                    # NB compute distance from this neighbor to nearest spot in clone.
                    min_dist_to_group = np.inf
                    for group_member in group:
                        dist_to_member = np.linalg.norm(
                            this_coords[neighbor] - this_coords[group_member]
                        )
                        min_dist_to_group = min(min_dist_to_group, dist_to_member)

                    # NB guard against disjoint groups.
                    # TODO tailor to Visium (HD).
                    if min_dist_to_group > 1.2 * last_dist:
                        break

                    if neighbor not in group:
                        group.add(neighbor)
                        group_umis += this_spot_counts[neighbor]

                        assigned[neighbor] = True

                        last_dist = min_dist_to_group

                    if group_umis >= min_clone_umis:
                        break

                num_rounds += 1

                if initial_group_umis == group_umis:
                    logger.warning(f"Saturated growth of current clone.")
                    break

                if num_rounds == max_growth_rounds:
                    logger.warning(
                        f"Max growth rounds={max_growth_rounds} reached for clone {clone_id} in sample {i}."
                    )
                    break

            # NB assign clone_id to these spots
            for g in group:
                assigned[g] = True
                clone_assignment[index[g]] = clone_id

            logger.info(f"Assigned {len(group)} spots to initial clone {clone_id}")

            clone_id += 1

    assert np.all(clone_assignment >= 0), "ERROR: spots were not assigned to a clone."

    clone_ids = np.unique(clone_assignment)

    logger.info(f"Solved for first pass at initial clones={clone_ids}")

    clone_total_umis = {}
    for cid in clone_ids:
        idx = np.where(clone_assignment == cid)[0]
        clone_total_umis[cid] = np.sum(spot_counts[idx])

    sufficient = np.array(
        [cid for cid in clone_ids if clone_total_umis[cid] >= min_clone_umis]
    )

    insufficient = np.array(
        [cid for cid in clone_ids if clone_total_umis[cid] < min_clone_umis]
    )

    if len(insufficient) > 0:
        logger.info(
            f"Found {len(insufficient)} clones with insufficient UMIs (< {min_clone_umis:_})."
        )

        if len(sufficient) == 0:
            fallback = max(clone_ids, key=lambda c: clone_total_umis[c])
            logger.warning(
                "No clone meets threshold; reassigning all insufficient spots to max-UMI clone."
            )
            for cid in insufficient:
                clone_assignment[clone_assignment == cid] = fallback
        else:
            suff_spot_mask = np.isin(clone_assignment, sufficient)
            suff_coords = coords[suff_spot_mask]
            suff_clone_ids = clone_assignment[suff_spot_mask]

            for cid in insufficient:
                insuff_spot_idxs = np.where(clone_assignment == cid)[0]

                for si in insuff_spot_idxs:
                    dists = np.linalg.norm(suff_coords - coords[si], axis=1)
                    target_clone = suff_clone_ids[np.argmin(dists)]
                    clone_assignment[si] = target_clone

        new_ids = sorted(np.unique(clone_assignment))
        id_map = {old: new for new, old in enumerate(new_ids)}
        for old, new in id_map.items():
            clone_assignment[clone_assignment == old] = new
        logger.info(
            f"After reassignment based on min. umi, number of clones={len(id_map)}."
        )

    initial_clone_index = [
        np.where(clone_assignment == i)[0] for i in range(np.max(clone_assignment) + 1)
    ]

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


# NB previously compute_adjacency_mat_v2
def anisotropic_distance_adjacency(coords, unit_xsquared=9, unit_ysquared=3, ratio=1):
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

    logger.info(
        f"Solving for distance-based adjacency matrix with ratio={ratio} and unit_xsquared={unit_xsquared}, unit_ysquared={unit_ysquared}"
    )

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


def anisotropic_exponential_decay_adjacency(
    coords, unit_xsquared=9, unit_ysquared=3, bandwidth=12, decay=5
):
    # NB x,y separations for all spot pairs.
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    # NB arbitrary normalized. y different than x!
    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    logger.info(
        f"Solving for slice Potts adjacency with exponential kernel based on squared distance with bandwidth,decay={bandwidth},{decay}."
    )

    kern = np.exp(-((pairwise_squared_dist / bandwidth) ** decay))

    # NB (spot, spot) adjacency.
    A = np.zeros((coords.shape[0], coords.shape[0]))

    for i in range(coords.shape[0]):
        indexes = np.where(kern[i, :] > 1e-4)[0]  # MAGIC
        indexes = np.array([j for j in indexes if j != i])

        if len(indexes) > 0:
            A[i, indexes] = kern[i, indexes]

    return scipy.sparse.csr_matrix(A)


def choose_lattice_adjacency(
    coords,
    single_total_bb_RD,
    maxspots_pooling=7,
    unit_xsquared=9,
    unit_ysquared=3,
    min_coordination_num=8,
):
    # NB called per slice.
    coordination_num = summarize_lattice_structure(
        coords, sample_ids=np.zeros(len(coords)), sample_list=[None]
    )

    if coordination_num < min_coordination_num:
        logger.warning(
            f"Assuming minimum coordination number={min_coordination_num}."
        )

        coordination_num = min_coordination_num

    logger.info(
        f"Assigning lattice adjacency matrix with coordination_num={coordination_num}, "
        f"assuming unit_xsquared,unit_ysquared={unit_xsquared},{unit_ysquared}."
    )

    n_spots = coords.shape[0]

    """
    x_dist = coords[:, 0][None, :] - coords[:, 0][:, None]
    y_dist = coords[:, 1][None, :] - coords[:, 1][:, None]

    pairwise_squared_dist = x_dist**2 * unit_xsquared + y_dist**2 * unit_ysquared

    # NB set diagonal to infinity to exclude self from nearest neighbors
    np.fill_diagonal(pairwise_squared_dist, np.max(pairwise_squared_dist))

    logger.info(f"Construcuted pairwise distances.")
    """

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

    logger.info(f"Assumed identity smooth mat.")

    """
    # NB adjacency matrix: connect each spot to coordination_num nearest neighbors
    A = np.zeros((n_spots, n_spots), dtype=np.float64)

    for i in range(n_spots):
        nearest_indices = np.argpartition(
            pairwise_squared_dist[i, :], coordination_num
        )[:coordination_num]

        if len(nearest_indices) > 0:
            A[i, nearest_indices] = 1.0

    adjacency_mat = scipy.sparse.csr_matrix(A)
    """

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
    # TODO np.inf, but integer.
    np.fill_diagonal(tmp_pairwise_squared_dist, np.max(tmp_pairwise_squared_dist))

    # NB given the minimum neighbor distance for all spots, find the median and normalize by the sum of scaling factors -
    #    used to set a baseline for neighborhood size.
    spot_min_distances = np.min(tmp_pairwise_squared_dist, axis=0)

    base_ratio = np.median(spot_min_distances) / (unit_xsquared + unit_ysquared)
    s_ratio = 0

    for ratio in range(10):
        # NB simple distance based adjacency: progressively greater separated spots are including as neighbors until
        #    the median neighbor count is > maxspots_pooling
        smooth_mat = anisotropic_distance_adjacency(
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

    # NB backtrack given we surpassed maxspots_pooling.
    smooth_mat = anisotropic_distance_adjacency(
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
        # NB distance-based kernel adjacency with assumed decay.
        adjacency_mat = anisotropic_exponential_decay_adjacency(
            coords, unit_xsquared, unit_ysquared, bandwidth=bandwidth
        )

        adjacency_mat.setdiag(1)

        # NB where smooth connection is stronger than exponential, we rely on smooth.
        #    runtime better with increased pooling.
        adjacency_mat = adjacency_mat - smooth_mat
        adjacency_mat[adjacency_mat < 0.0] = 0.0

        # NB we expect a coordination number of 6 on a hexagonal lattice.  MAGIC?
        if np.median(np.sum(adjacency_mat, axis=0).A.flatten()) >= 6:
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


@cacher("adjacency.hdf5")
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
    logger.info("Solving for multi-slice adjacency (and spot-pooling) matrix.")

    # NB smooth_mat contains the edges of spots that are directly pooled.
    adjacency_mat, smooth_mat = [], []

    # NB loop over slices.
    for i, _ in enumerate(sample_list):
        # NB spots per slice.
        index = np.where(sample_ids == i)[0]

        # NB (x,y) for these spots.
        this_coords = np.array(coords[index, :])

        """
        tmpsmooth_mat, tmpadjacency_mat = choose_adjacency_by_readcounts(
            this_coords,
            single_total_bb_RD[:, index],
            maxspots_pooling=maxspots_pooling,
            unit_xsquared=unit_xsquared,
            unit_ysquared=unit_ysquared,
        )
        """
        
        tmpsmooth_mat, tmpadjacency_mat = choose_lattice_adjacency(
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

    logger.info("Solving for multi-slice adjacency (and spot-pooling) matrix.")

    Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])

    return Adjacency(adjacency_mat=adjacency_mat, smooth_mat=smooth_mat)
