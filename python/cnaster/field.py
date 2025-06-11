import numpy as np


def planar_power_law(positions, k, exponent, scale=2.0):
    modk = np.linalg.norm(k)
    scalars = np.dot(positions, k) / modk

    field = (scalars - scale * exponent) ** exponent
    field[scalars < 0.0] = 0.0
    field[scalars - scale * exponent < 0.0] = 0.0

    return field


def planar_power_law_fields(
    positions, max_label, vec_k, scale=2.0, xlower=10.0, ylower=10.0
):
    exponents = np.array([0.0, 1.0, 1.25])

    fields = np.vstack(
        [planar_power_law(positions, vec_k, exp, scale=scale) for exp in exponents]
    )

    isin = positions[:, 0] > xlower
    isin &= positions[:, 1] > ylower

    fields[:, ~isin] = 0.0

    return fields.T
