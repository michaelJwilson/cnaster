import numpy as np


class Ellipse:
    def __init__(self, center, L):
        self.center = np.asarray(center, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.Q = self.L @ self.L.T
        self.n = 2
        self.det_L = np.linalg.det(self.L)

    def __repr__(self):
        return (
            f"Ellipse(\n"
            f"  center={self.center.tolist()},\n"
            f"  det_L={self.det_L:.4g},\n"
            f"  L=\n{np.array2string(self.L, precision=4, suppress_small=True)}\n"
            f")"
        )

    @classmethod
    def from_diagonal(cls, center, diag):
        L = np.diag(1.0 / np.array(diag, dtype=float))

        return cls(center, L)

    @classmethod
    def from_linear(cls, center, L):
        return cls(center, L)

    def contains(self, pos_xy):
        v = np.array(pos_xy) - self.center

        return np.linalg.norm(self.L.T @ v) ** 2 <= 1.0

    def rotate(self, theta):
        """
        theta: rotation angle in radians
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        return Ellipse.from_linear(self.center, R @ self.L)

    def get_daughter(self, factor):
        # NB parent corresponds to the unit ball pushed forward by L.
        new_radius = factor * 1.0

        new_x = new_radius - 1.0 + self.center[0]
        new_center = np.array([new_x, self.center[1]])

        # NB new projection
        theta = (np.pi / 4.0) * np.random.uniform()
        
        new_L = self.rotate(theta).L
        new_L /= factor**2.

        # NB see https://arxiv.org/pdf/1908.09326
        return Ellipse.from_linear(new_center, new_L)
