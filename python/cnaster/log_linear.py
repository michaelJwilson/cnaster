import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms, scale as mscale, ticker as mticker


class LinearThenLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1
    is_separable = True

    def __init__(self, threshold=1.0, base=10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.base = float(base)

    def transform_non_affine(self, a):
        a = np.asanyarray(a, dtype=float)
        out = np.empty_like(a, dtype=float)
        mask = a <= self.threshold
        out[mask] = a[mask]
        # map values > threshold to threshold + log_base(y/threshold)
        out[~mask] = self.threshold + np.log(a[~mask] / self.threshold) / np.log(
            self.base
        )
        return out

    def inverted(self):
        return InvertedLinearThenLogTransform(self.threshold, self.base)


class InvertedLinearThenLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1
    is_separable = True

    def __init__(self, threshold=1.0, base=10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.base = float(base)

    def transform_non_affine(self, a):
        a = np.asanyarray(a, dtype=float)
        out = np.empty_like(a, dtype=float)
        mask = a <= self.threshold
        out[mask] = a[mask]
        # inverse: y = threshold * base^(v - threshold)
        out[~mask] = self.threshold * (self.base ** (a[~mask] - self.threshold))
        return out

    def inverted(self):
        return LinearThenLogTransform(self.threshold, self.base)


class LinearThenLogScale(mscale.ScaleBase):
    name = "linlog"

    def __init__(self, axis, *, threshold=1.0, base=10.0):
        super().__init__(axis)
        self.threshold = float(threshold)
        self.base = float(base)

    def get_transform(self):
        return LinearThenLogTransform(self.threshold, self.base)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.AutoLocator())
        axis.set_minor_locator(mticker.AutoMinorLocator())
        axis.set_major_formatter(mticker.ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

mscale.register_scale(LinearThenLogScale)
