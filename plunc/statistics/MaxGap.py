import numpy as np
from .base import TestStatistic
from scipy import stats as scipy_stats


class MaxGap(TestStatistic):
    """The Yellin maximum gap statistic -- inverted (so it is more proportional to mu)
    This uses Monte Carlo training rather than Yellin's exact equation (2 in his paper)
    as the exact equation will cause overflow errors at very high n
    """
    stats = np.linspace(0, 100, 1000 + 1)
    mus = np.linspace(0, 100, 1000 + 1)
    statistic_name = 'ginv'
    distribution = scipy_stats.uniform
    n_training_trials = 1000

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

    def event_generator(self, n_trials):
        return np.random.uniform(0, 1, n_trials)