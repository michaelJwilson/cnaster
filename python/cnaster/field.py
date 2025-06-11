import numpy as np


def planar_power_law(planar_positions, k, exponent):
    modk = np.linalg.norm(k)
    modx = np.sqrt(planar_positions[:, 0] ** 2.0 + planar_positions[:, 1] ** 2.0)

    scalars = np.dot(planar_positions, k) / modk
    scalars /= scalars.max()

    return scalars**exponent


def planar_power_law_fields(positions, max_label, vec_k, epsilon=0.25):
    exponents = np.array([1.0 - (n + 1) * epsilon for n in range(1 + max_label)])

    planar_positions = positions[:, :-1]

    fields = np.vstack(
        [planar_power_law(planar_positions, vec_k, exp) for exp in exponents]
    )

    return fields.T
