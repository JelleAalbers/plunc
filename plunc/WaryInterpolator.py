import logging

import numpy as np

from plunc.exceptions import InsufficientPrecisionError, OutsideDomainError


class WaryInterpolator(object):
    """Interpolate (and optionally extrapolation) between points,
    raising exception if error larger than desired precision
    """

    def __init__(self,
                 points=tuple(), values=tuple(),
                 precision=0.01, domain=(-1, 3),
                 if_lower='raise', if_higher='extrapolate',):
        """
        :param points:
        :param values:
        :param precision:
        :param domain: (low, high) boundaries of region where interpolation is used
                       If no values are known at the boundaries, the effective boundary is tighter
        :param if_lower: 'extrapolate' or 'raise'
        :param if_higher:
        :param loglevel:
        :return:
        """
        self.precision = precision
        self.domain = domain
        self.if_lower = if_lower
        self.if_higher = if_higher
        self.log = logging.getLogger('WaryInterpolator')

        self.points = np.array(points)
        self.values = np.array(values)

    def __call__(self, x):
        self.log.debug("Asked for x = %s" % x)
        if len(self.points) < 3:
            raise InsufficientPrecisionError("Need at least three datapoints before we can interpolate or extrapolate")

        if x < self.domain[0]:
            self.log.debug("Below domain boundary")
            if self.if_lower == 'extrapolate':
                return self.extrapolate(x)
            else:
                raise OutsideDomainError("Value %s is below the lowest known value %s" % (x, self.points.min()))

        elif x > self.domain[1]:
            self.log.debug("Above domain boundary")
            if self.if_higher == 'extrapolate':
                return self.extrapolate(x)
            else:
                raise OutsideDomainError("Value %s is above the highest known value %s" % (x, self.points.max()))

        else:
            return self.interpolate(x)

    def interpolate(self, x):
        if x in self.points:
            self.log.debug("Exact value known")
            return self.values[np.nonzero(self.points == x)[0][0]]

        max_i = len(self.points) - 1

        if not self.points.min() < x < self.points.max():
            self.log.debug("%s is in domain, but outside the range of known values. Trying extrapolation." % x)
            return self.extrapolate(x)

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
            raise InsufficientPrecisionError("Interpolation failed: achieved precision %s, required %s" % (
                abs(diff/y), self.precision))

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
        if x > self.domain[0]:
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

    def add_point(self, x, y):
        self.add_points(np.array([x]), np.array([y]))

    def add_points(self, xs, ys):
        if not self.domain[0] <= np.min(xs) <= np.max(xs) <= self.domain[1]:
            raise ValueError("Points to add must lie in the domain [%s-%s], but you passed values from %s to %s" % (
                self.domain[0], self.domain[1], np.min(xs), np.max(xs)))
        self.points = np.concatenate((self.points, xs))
        self.values = np.concatenate((self.values, ys))
        sort_indices = np.argsort(self.points)
        self.points = self.points[sort_indices]
        self.values = self.values[sort_indices]

    def plot(self):
        if not self.loglog_space:
            raise NotImplementedError
        import matplotlib.pyplot as plt
        x = np.logspace(self.domain[0], self.domain[1], 100)
        plt.plot(np.log10(x), [np.log10(self.f(q)) for q in x])
        plt.plot(self.points, self.values, marker='o')