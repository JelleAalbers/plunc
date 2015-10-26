import numpy as np
from scipy import stats

from .base import TestStatistic


class NumberOfEvents(TestStatistic):
    """The most basic test statistic: the number of events observed.
    This is one of few test statistics that does not need Monte Carlo training
    """
    statistic_name = 'n'
    auto_train = False

    def __call__(self, observation, hypothesis=None):
        return len(observation)

    def likelihood(self, values, mu):
        # If the hypothesis is 0, return 1 for 0 events, 0 otherwise
        if mu == 0:
            return (np.sign(values) + 1) % 2
        return stats.poisson.pmf(values, mu=mu)

    def likelihood_leq(self, values, mu):
        # If the hypothesis is 0, return 1 for 0 events, 0 otherwise
        if mu == 0:
            return (np.sign(values) + 1) % 2
        return stats.poisson.cdf(values, mu=mu)