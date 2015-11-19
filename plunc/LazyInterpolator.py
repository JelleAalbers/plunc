import numpy as np

from plunc.exceptions import InsufficientPrecisionError


class LazyInterpolator(object):
    """Use interpolation (and optionally extrapolation) if possible, call function if precision requires it.
    When function is called, value is stored and improves interpolation in the future.
    """

    def __init__(self, f, precision=0.01, operating_range=(-1, 3),
                 if_lower='exact', if_higher='extrapolate', loglog_space=True):
        self.f = f
        self.precision = precision
        self.operating_range = operating_range
        self.if_lower = if_lower
        self.if_higher = if_higher
        self.loglog_space = loglog_space

        self.points = np.array([])
        self.values = np.array([])

        # Pre-calculate the values at the boundaries and midpoint
        # This is needed to have enough points for interpolation and extrapolation
        self(operating_range[0])
        self(operating_range[1])
        self(operating_range[1] + operating_range[0])

    def __call__(self, x):
        if self.loglog_space:
            x = np.log10(x)

        if x < self.operating_range[0]:
            if self.if_lower == 'exact':
                y = self.f(x)
            elif self.if_lower == 'extrapolate':
                y = self.extrapolate(x)
            else:
                raise ValueError("Value %s is below the operational range "
                                 "lower bound %s" % (x, self.operating_range[0]))

        elif x > self.operating_range[1]:
            if self.if_higher == 'exact':
                y = self.f(x)
            elif self.if_higher == 'extrapolate':
                y = self.extrapolate(x)
            else:
                raise ValueError("Value %s is above the operational range "
                                 "upper bound %s" % (x, self.operating_range[0]))

        else:
            if x in self.points:
                y = self.values[np.nonzero(self.points == x)[0][0]]
            else:
                y = self.interpolate(self, x)

        if self.loglog_space:
            return 10**y
        return y

    def interpolate(self, x):
        max_i = len(self.points) - 1

        # Find index of nearest known point to the right
        nearest_right = np.searchsorted(self.points, x)
        assert 0 < nearest_right < len(self.points)

        if nearest_right == 1:
            y = self.linear_interpolant(x, 0, 1)
            y2 = self.linear_interpolant(x, 0, 2)
            diff = 2 * (y - y2)
        elif nearest_right == max_i:
            y = self.linear_interpolant(x, max_i - 1, max_i)
            y2 = self.linear_interpolant(x, max_i - 2, max_i)
            diff = 2 * (y - y2)
        else:
            y = self.linear_interpolant(x, nearest_right - 1, nearest_right)
            y2 = self.linear_interpolant(x, nearest_right - 1, nearest_right + 1)
            diff = y - y2

        if abs(diff / y) > self.precision:
            return self.call_and_add(x)
        return y

    def linear_interpolant(self, x, index_low, index_high):
        y0, y1 = self.values[(index_low, index_high)]
        x0, x1 = self.points[(index_low, index_high)]
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
        if abs(diff / y) > self.precision:
            raise InsufficientPrecisionError("Extrapolation precision %s estimated, "
                                             "but %s required" % (abs(diff / y), self.precision))
        return y

    def call_and_add(self, x):
        """Returns self.f(x), storing its value for the benefit of future interpolation"""
        y = self.f(x)
        # Add the point, then re-sort the points & values arrays
        # ... very inefficient use of arrays ...
        assert self.operating_range[0] <= x <= self.operating_range[1]
        self.points = np.concatenate((self.points, [x]))
        self.values = np.concatenate((self.points, [y]))
        sort_indices = np.argsort(self.points)
        self.points = self.points[sort_indices]
        self.values = self.values[sort_indices]
        self.recalculate_extrapolation = True
        return y
