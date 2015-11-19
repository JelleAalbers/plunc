import numpy as np
import logging


from plunc.exceptions import InsufficientPrecisionError


class LazyInterpolator(object):
    """Use interpolation (and optionally extrapolation) if possible, call function if precision requires it.
    When function is called, value is stored and improves interpolation in the future.
    """

    def __init__(self, f, precision=0.01, operating_range=(-1, 3),
                 if_lower='exact', if_higher='extrapolate', loglog_space=True, loglevel='INFO'):
        self.f = f
        self.precision = precision
        self.operating_range = operating_range
        self.if_lower = if_lower
        self.if_higher = if_higher
        self.loglog_space = loglog_space
        self.log = logging.getLogger('LazyInterpolator')
        self.log.setLevel(getattr(logging, loglevel.upper()))

        self.points = np.array([])
        self.values = np.array([])

        # Pre-calculate the values at the boundaries and midpoint
        # This is needed to have enough points for interpolation and extrapolation
        for x in [operating_range[0], operating_range[1], 0.5 * (operating_range[0] + operating_range[1])]:
            if self.loglog_space:
                self(10**x)
            else:
                self(x)

    def __call__(self, x, force_call=False, **extra_kwargs):
        """
        force_call:      Force f to be called no matter what.
        Other kwargs will be passed to f, if it has to be called
        """
        self.log.debug("Asked for x = %s" % x)
        if self.loglog_space:
            x = np.log10(x)

        if x < self.operating_range[0]:
            self.log.debug("Below operating range")
            if self.if_lower == 'exact' or force_call:
                self.log.debug("Calling f without storing the value")
                y = self.f(x, **extra_kwargs)
            elif self.if_lower == 'extrapolate':
                y = self.extrapolate(x)
            else:
                raise ValueError("Value %s is below the operational range "
                                 "lower bound %s" % (x, self.operating_range[0]))

        elif x > self.operating_range[1]:
            self.log.debug("Above operating range")
            if self.if_higher == 'exact' or force_call:
                self.log.debug("Calling f without storing the value")
                y = self.f(x, **extra_kwargs)
            elif self.if_higher == 'extrapolate':
                y = self.extrapolate(x)
            else:
                raise ValueError("Value %s is above the operational range "
                                 "upper bound %s" % (x, self.operating_range[0]))

        else:
            if force_call:
                self.log.debug("Calling f by force (in range, so can store value)")
                y = self.call_and_add(x, **extra_kwargs)
            elif x in self.points:
                self.log.debug("Exact value known")
                y = self.values[np.nonzero(self.points == x)[0][0]]
            else:
                y = self.interpolate_or_call(x, **extra_kwargs)

        self.log.debug("Returning y = %s" % y)
        if self.loglog_space:
            y = 10**y
        return y

    def interpolate_or_call(self, x, **extra_kwargs):
        self.log.debug("In range, trying to call or interpolate")
        max_i = len(self.points) - 1
        if max_i < 2:
            self.log.debug("Call because of insufficient training")
            # We need at least three points for interpolation, these are added at the start
            return self.call_and_add(x)

        # Find index of nearest known point to the right
        nearest_right = np.searchsorted(self.points, x)
        assert 0 < nearest_right < len(self.points)

        if nearest_right == 1:
            self.log.debug("Only one point to left")
            y = self.linear_interpolant(x, 0, 1)
            y2 = self.linear_interpolant(x, 0, 2)
            diff = 2 * (y - y2)
        elif nearest_right == max_i:
            self.log.debug("Only one point to right")
            y = self.linear_interpolant(x, max_i - 1, max_i)
            y2 = self.linear_interpolant(x, max_i - 2, max_i)
            diff = 2 * (y - y2)
        else:
            self.log.debug("At least two points on either side")
            y = self.linear_interpolant(x, nearest_right - 1, nearest_right)
            y2 = self.linear_interpolant(x, nearest_right - 1, nearest_right + 1)
            diff = y - y2

        self.log.debug("Close interpolation gives y=%s, far gives y=%s.\n"
                       "Difference factor %s, precision tolerance %s" % (y, y2, abs(diff / y), self.precision))

        if abs(diff / y) > self.precision:
            self.log.debug("Calling because of insufficient precision in interpolation")
            return self.call_and_add(x, **extra_kwargs)
        self.log.debug("Interpolation is ok, returning result")
        return y

    def linear_interpolant(self, x, index_low, index_high):
        x0 = self.points[index_low]
        x1 = self.points[index_high]
        y0 = self.values[index_low]
        y1 = self.values[index_high]
        return y0 + (y1 - y0) * (x - x0)/(x1 - x0)

    def extrapolate(self, x):
        # TODO: change to linear regression on all points in configurable part (e.g. 5%) of range (for y2, 2x range)
        # Now this is very vulnerable to small errors on datapoints if datapoints are close together near edge
        # TODO: option to ignore InsufficientPrecisionError and barge ahead anyway
        if x > self.operating_range[0]:
            max_i = len(self.points) - 1
            y = self.linear_interpolant(x, max_i - 1, max_i)
            y2 = self.linear_interpolant(x, max_i - 2, max_i)
        else:
            y = self.linear_interpolant(x, 0, 1)
            y2 = self.linear_interpolant(x, 0, 2)

        diff = 2 * (y - y2)
        self.log.debug("Close extrapolation gives y=%s, far gives y=%s.\n"
                       "Difference factor %s, precision tolerance %s" % (y, y2, abs(diff / y), self.precision))

        if abs(diff / y) > self.precision:
            raise InsufficientPrecisionError("Extrapolation precision %s estimated, "
                                             "but %s required" % (abs(diff / y), self.precision))
        return y

    def call_and_add(self, x, **extra_kwargs):
        """Returns self.f(x), storing its value for the benefit of future interpolation"""
        if self.loglog_space:
            y = np.log10(self.f(10**x, **extra_kwargs))
        else:
            y = self.f(x, **extra_kwargs)
        # Add the point, then re-sort the points & values arrays
        # ... very inefficient use of arrays ...
        if not self.operating_range[0] <= x <= self.operating_range[1]:
            raise RuntimeError("Cannot add point %s, ouside of operating range [%s, %s]!" % (x,
                                                                                             self.operating_range[0],
                                                                                             self.operating_range[1]))
        self.points = np.concatenate((self.points, [x]))
        self.values = np.concatenate((self.values, [y]))
        sort_indices = np.argsort(self.points)
        self.points = self.points[sort_indices]
        self.values = self.values[sort_indices]
        self.recalculate_extrapolation = True
        return y

    def plot(self):
        if not self.loglog_space:
            raise NotImplementedError
        import matplotlib.pyplot as plt
        x = np.logspace(self.operating_range[0], self.operating_range[1], 100)
        plt.plot(np.log10(x), [np.log10(self.f(q)) for q in x])
        plt.plot(self.points, self.values, marker='o')