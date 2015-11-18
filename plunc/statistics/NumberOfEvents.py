import numpy as np
from scipy import stats

from .base import TestStatistic
from plunc.common import round_to_digits


class NumberOfEvents(TestStatistic):
    """The most basic test statistic: the number of events observed.
    This is one of few test statistics that does not need Monte Carlo training
    """
    statistic_name = 'n'

    def __call__(self, observation, hypothesis=None):
        return len(observation)

    def get_values_and_likelihoods(self, mu, precision_digits=None):
        if mu == 0:
            # Return two possible values: 0 and 1, with probability 1 and 0
            # There must be more than one possible value to avoid problems
            # e.g. upper limit should return infinite interval, lower limit single point at 0
            # for this we need to check if 1 is included or not
            return np.array([0, 1]), np.array([1, 0])
        # Step size
        step_size = max(1, 10 **(int(np.log10(mu)) - precision_digits))
        max_mu = round_to_digits(4 + mu + 4 * np.sqrt(mu), precision_digits)
        values = np.arange(0, max_mu, step_size)
        return values, stats.poisson.pmf(values, mu=mu)

    def probability(self, values, mu):
        # If the hypothesis is 0, return 1 for 0 events, 0 otherwise
        if mu == 0:
            return (np.sign(values) + 1) % 2
        return stats.poisson.pmf(values, mu=mu)

    def probability_leq(self, values, mu):
        # If the hypothesis is 0, return 1 for 0 events, 0 otherwise
        if mu == 0:
            return (np.sign(values) + 1) % 2
        return stats.poisson.cdf(values, mu=mu)