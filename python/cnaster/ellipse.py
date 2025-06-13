import numpy as np


class Ellipse:
    def __init__(self, center, L):
        self.center = np.asarray(center, dtype=float)
        self.L = np.asarray(L, dtype=float)
        self.Q = self.L @ self.L.T
        self.n = 2
        self.det_L = np.linalg.det(self.L)

    @classmethod
    def from_diagonal(cls, center, diag):
        L = np.diag(1.0 / np.array(diag, dtype=float))

        return cls(center, L)

    @classmethod
    def from_linear(cls, center, L):
        return cls(center, L)

    def contains(self, pos_xy):
        v = np.array(pos_xy) - self.center

        return np.linalg.norm(L.T @ v) ** 2 <= 1.0

    def rotate(theta):
        """
        theta: rotation angle in radians
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        return R @ self.L

    def get_daughter(self, radius_factor):
        # NB parent corresponds to the unit ball pushed forward by L
        #
        new_radius = factor * 1.0

        new_x = new_radius - 1.0 + self.center[0]
        new_center = np.array([new_x, self.center[1]])

        # NB new projection
        theta = (np.pi / 4.0) * np.random.uniform()
        new_L = rotate(theta)

        return Ellipse.from_linear(new_center, new_L)
