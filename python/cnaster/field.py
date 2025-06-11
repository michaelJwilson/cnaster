import numpy as np


def planar_power_law(planar_positions, k, exponent):
    k = np.asarray(k)
    k_sq = np.dot(k, k)

    scalars = np.dot(planar_positions, k) / k_sq
    
    return scalars


def planar_power_law_fields(positions, max_label, vec_k):
    modk = np.linalg.norm(vec_k)

    wavelength = (2.0 * np.pi) / modk

    exponents = np.random.uniform(size=(1 + max_label))
    exponents /= exponents.max()
    exponents = np.sort(exponents)

    print(exponents)
    
    planar_positions = positions[:, :-1]

    fields = np.vstack(
        [planar_power_law(planar_positions, vec_k, exp) for exp in exponents]
    )

    return fields.T
