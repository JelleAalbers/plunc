"""
Basic interval choice methods like upper and lower limit
"""
from .base import IntervalChoice
import numpy as np


class UpperLimit(IntervalChoice):

    def rank_stat_values(self, **kwargs):
        return -kwargs['statistic_values']


class LowerLimit(IntervalChoice):

    def rank_stat_values(self, **kwargs):
        return kwargs['statistic_values']

class MinimumLength(IntervalChoice):
    """Add points in order of likelihood - Crow & Gardner method
    Note this allows the interval to become non-central:
    i.e. it can become more probable for the value to lie above the interval than to lie below it
    """
    def rank_stat_values(self, **kwargs):
        return np.abs(kwargs['likelihoods'] - 0.5)

class CentralCI(IntervalChoice):
    """Require there is never more than (1 - confidence_level)/2 probability mass on either side of the CI
    We'll rank the the values by min(pleft, pright),
        where pleft = p mass at <= value, p right likewise
    then enforce a threshold for inclusion (rather than a ranking algorithm)

    ... surprised this is so complicated!
    """
    method = 'threshold'

    def get_threshold(self):
        return (1 - self.cl) / 2

    def rank_stat_values(self, **kwargs):
        p = kwargs['likelihoods']
        p_right = np.cumsum(p)
        p_left = 1 - p_right + p
        # Why doesn't have numpy have a straightforward elementwise min for two arrays??
        return np.clip(p_left, 0, p_right)
