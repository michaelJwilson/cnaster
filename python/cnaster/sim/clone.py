import json
import numpy as np
from cnaster_rs import ellipse


class Clone:
    def __init__(self, fpath, x0=None):
        with open(fpath, "r") as f:
            data = json.load(f)

        center = np.array(data["center"]).reshape(2, 1)
        L = np.array(data["L"])

        if x0 is not None:
            x0 = x0.reshape.reshape(2, 1)
            center += x0

        self.id = data["id"]
        self.cnas = data["cnas"]
        self.ellipse = ellipse.CnaEllipse(center, L)

    def __repr__(self):
        return f"<Clone ellipse={self.ellipse} cnas={len(self.cnas)}>"
