import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

Adjacency = namedtuple("Adjacency", ["adjacency_mat", "smooth_mat"])


def multislice_adjacency(
    coords,
    sample_ids,
    lattice_type=None,
    n_nearest=6,
    dx=1.0,
    dy=1.0,
    across_slice_adjacency_mat=None,
):
    """
    Efficiently builds a sparse multi-slice adjacency graph based on scaled nearest neighbors.
    If lattice_type is provided ('square', 'triangular', 'hexagonal'), it automatically
    overrides n_nearest with the mathematical coordination number.
    """
    # 1. Resolve coordination number based on lattice geometry
    lattice_map = {
        "square": 4,  # e.g., Visium HD, Stereo-seq
        "hexagonal": 3,  # True honeycomb
        "triangular": 6,  # Staggered packing (Standard Visium v1)
        "visium": 6,  # Convenience alias for Standard Visium
    }

    if lattice_type is not None:
        l_type = str(lattice_type).lower()
        if l_type in lattice_map:
            n_nearest = lattice_map[l_type]
            logger.info(
                f"Lattice type '{l_type}' detected. Overriding to n_nearest={n_nearest}."
            )
        else:
            logger.warning(
                f"Unknown lattice_type '{lattice_type}'. Falling back to n_nearest={n_nearest}."
            )
    else:
        logger.info(f"No lattice_type provided. Using fallback n_nearest={n_nearest}.")

    n_spots = coords.shape[0]

    # 2. Smooth Mat: Purely diagonal (identity matrix), each spot pools only itself
    smooth_mat = scipy.sparse.identity(n_spots, dtype=np.float64, format="csr")

    # 3. Build Adjacency Mat slice-by-slice
    adj_blocks = []
    num_slices = int(np.max(sample_ids)) + 1

    for s in range(num_slices):
        idx = np.where(sample_ids == s)[0]
        n_slice_spots = len(idx)

        if n_slice_spots == 0:
            adj_blocks.append(scipy.sparse.csr_matrix((0, 0)))
            continue

        slice_coords = coords[idx].copy().astype(float)
        slice_coords[:, 0] *= dx
        slice_coords[:, 1] *= dy

        tree = cKDTree(slice_coords)

        k_neighbors = min(n_nearest + 1, n_slice_spots)
        _, neighbors = tree.query(slice_coords, k=k_neighbors)

        neighbors = neighbors[:, 1:]

        rows = np.repeat(np.arange(n_slice_spots), k_neighbors - 1)
        cols = neighbors.flatten()
        data = np.ones(len(rows), dtype=np.float64)

        slice_adj = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_slice_spots, n_slice_spots)
        )

        slice_adj = slice_adj.maximum(slice_adj.T)
        adj_blocks.append(slice_adj)

    # 4. Combine slice blocks diagonally
    adjacency_mat = scipy.sparse.block_diag(adj_blocks, format="csr")

    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    adjacency_mat.setdiag(0)
    adjacency_mat.eliminate_zeros()

    logger.info(
        f"Adjacency graph complete: {adjacency_mat.nnz} total edges across {num_slices} slices."
    )

    # return Adjacency(adjacency_mat=adjacency_mat, smooth_mat=smooth_mat)

    return adjacency_mat, smooth_mat
