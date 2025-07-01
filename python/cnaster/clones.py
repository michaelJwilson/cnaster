import numpy as np

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


def initialize_clones(coords, sample_ids, x_part, y_part, single_tumor_prop, threshold):
    initial_clone_index = []

    for s in range(np.max(sample_ids) + 1):
        index = np.where(sample_ids == s)[0]

        assert len(index) > 0

        tmp_clone_index = fixed_rectangle_partition(
            coords[index, :],
            x_part,
            y_part,
            single_tumor_prop[index],
            threshold=threshold,
        )

        for x in tmp_clone_index:
            initial_clone_index.append(index[x])

    return initial_clone_index
