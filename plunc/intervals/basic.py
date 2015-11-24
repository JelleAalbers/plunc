"""
Basic interval choice methods like upper and lower limit
"""
from .base import IntervalChoice
import numpy as np


class UpperLimit(IntervalChoice):
    fixed_lower_limit = 0

    def score_stat_values(self, **kwargs):
        return -kwargs['statistic_values']


class LowerLimit(IntervalChoice):
    fixed_upper_limit = 0

    def score_stat_values(self, **kwargs):
        return kwargs['statistic_values']

class MinimumLength(IntervalChoice):
    """Add points in order of likelihood (Crow & Gardner method?)
    Note this usually results in a non-central confidence interval:
    i.e. it can become more probable for the value to lie above the interval than to lie below it
    """
    def score_stat_values(self, **kwargs):
        return np.abs(kwargs['likelihoods'] - 0.5)

class CentralCI(IntervalChoice):
    """Require there is never more than (1 - confidence_level)/2 probability mass on either side of the CI
    We'll rank the the values by min(pleft, pright),
        where pleft = p mass at <= value, p right likewise
    then enforce a threshold for inclusion.
    We can't use the ranking algorithm: the confidence level does not (directly) constrain the total likelihood
    contained in the interval, as in almost all other methods.
    ... surprised this is so complicated!
    """
    method = 'threshold'

    def get_threshold(self):
        return (1 - self.cl) / 2

    def score_stat_values(self, **kwargs):
        p = kwargs['likelihoods']
        p_right = np.cumsum(p)
        p_left = 1 - p_right + p
        # Why doesn't have numpy have a straightforward elementwise min for two arrays??
        return np.clip(p_left, 0, p_right)
