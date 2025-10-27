import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtransforms, scale as mscale, ticker as mticker
import matplotlib.ticker as mticker

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

class LinearThenLogMinorLocator(mticker.Locator):
    """Minor locator: auto below threshold, log-like above threshold."""
    def __init__(self, threshold=1.0, base=10.0):
        self.threshold = threshold
        self.base = base

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if vmax <= self.threshold:
            # entirely linear region -> no minor ticks (or use AutoMinorLocator)
            return []
        # above threshold: use log minor ticks
        # map back to data space
        transform = self.axis.get_transform()
        inv = transform.inverted()
        # get log-spaced minors in transformed space
        t_min = max(self.threshold, vmin)
        t_max = vmax
        # simple approach: log minors from threshold to vmax
        log_min = np.log(t_min / self.threshold) / np.log(self.base)
        log_max = np.log(t_max / self.threshold) / np.log(self.base)
        decades = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
        ticks = self.threshold * (self.base ** decades)
        # filter to view range
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        return ticks


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
        axis.set_minor_locator(LinearThenLogMinorLocator(self.threshold, self.base))
        axis.set_major_formatter(mticker.ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

mscale.register_scale(LinearThenLogScale)
