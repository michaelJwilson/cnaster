import numpy as np


def planar_power_law(planar_positions, k, exponent):
    modk = np.linalg.norm(k)
    scalars = np.dot(planar_positions, k) / modk

    return scalars **exponent


def planar_power_law_fields(positions, max_label, vec_k):
    exponents = np.array([n for n in range(1 + max_label)])
    planar_positions = positions[:, :-1]

    # print(exponents)
    
    fields = np.vstack(
        [planar_power_law(planar_positions, vec_k, exp) for exp in exponents]
    )

    return fields.T
