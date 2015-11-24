import numpy as np
from .base import TestStatistic
from scipy import stats as scipy_stats


class MaxGap(TestStatistic):
    """The Yellin maximum gap statistic -- inverted (so it is more proportional to mu)
    This uses Monte Carlo training rather than Yellin's exact equation (2 in his paper)
    as the exact equation will cause overflow errors at very high n
    """
    statistic_name = 'ginv'
    distribution = scipy_stats.uniform

    def __call__(self, observation, hypothesis=None):
        n = len(observation)
        if n == 0:
            gap = 1
        else:
            transformed_obs = self.distribution.cdf(observation)
            transformed_obs.sort()
            if n == 1:
                gap = max(transformed_obs[0], 1 - transformed_obs[0])
            else:
                gap = max(transformed_obs[0],
                          np.diff(transformed_obs).max(),
                          1 - transformed_obs[-1])
        return 1/gap

    def get_values_and_likelihoods(self, mu, precision_digits=3):
        if mu == 0:
            # See NumberOfEvents on why you must always return more than one 'possible' value
            return np.array([1, 2]), np.array([1, 0])
        return TestStatistic.get_values_and_likelihoods(self, mu, precision_digits)

    def generate_single_observation(self, n_trials):
        return self.distribution.rvs(size=n_trials)