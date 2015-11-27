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


from plunc.intervals.basic import UpperLimit
from plunc.common import round_to_digits

class BinnedOptimumInterval(TestStatistic):

    statistic_name = 'iinv'
    distribution = scipy_stats.uniform

    def __init__(self, *args, n_bins=10, **kwargs):
        self.n_bins = n_bins
        TestStatistic.__init__(self, *args, **kwargs)
        self.cached_obss = dict()

        # Create the child statistics
        self.child_statistics = []
        for interval_size in range(self.n_bins):
            self.child_statistics.append(UpperLimit(
                statistic=BinnedOptimumInterval_OneIntervalSize(parent=self,
                                                                interval_size=interval_size),
                interpolator_log_limits=(-1, 2)))

    def observation_to_n_sparsest(self, observation):
        transformed_obs = self.distribution.cdf(observation)
        binned_obs, _ = np.histogram(transformed_obs, bins=self.n_bins, range=(0, 1))
        binned_obs_cumsum = np.cumsum(binned_obs)
        n_sparsest = np.zeros(self.n_bins, dtype=np.int)
        for interval_size in range(self.n_bins):
            # Interval size 0: 1 bin.
            # How many events are in the sparsest interval of interval_size bins?
            if interval_size == self.n_bins - 1:
                n_sparsest[interval_size] = len(transformed_obs)
            else:
                n_sparsest[interval_size] = (binned_obs_cumsum[(1+interval_size):] -
                                             binned_obs_cumsum[:-1*(1+interval_size)]).min()
        return n_sparsest

    def get_values_and_likelihoods_for_interval_size(self, interval_size, mu):
        # print("Getting v&l for interval size %s, mu=%s" % (interval_size, mu))
        if mu not in self.cached_obss:
            obss = self.generate_observations(mu=mu, n_trials=self.n_trials)
            assert len(obss) == self.n_trials
            self.cached_obss[mu] = np.zeros((self.n_trials, self.n_bins), dtype=np.int)
            for obs_i, obs in enumerate(obss):
                self.cached_obss[mu][obs_i, :] = self.observation_to_n_sparsest(obs)
        obs_of_this_size = self.cached_obss[mu][:, interval_size]
        counts = np.bincount(obs_of_this_size)
        return np.arange(len(counts)), counts/len(obs_of_this_size)

    def __call__(self, observation, hypothesis=None):
        n_sparsest = self.observation_to_n_sparsest(observation)
        child_limits = np.zeros(self.n_bins)
        precision_digits = self.child_statistics[0].precision_digits
        child_limit_search_region = [0, round_to_digits(10 + 6 * len(observation),
                                                        precision_digits)]
        for interval_size in range(self.n_bins):
            # print("Getting child limit for seeing %d events in interval size %d" % (n_sparsest[interval_size],
            #                                                                         interval_size))
            child_limits[interval_size] = self.child_statistics[interval_size].get_confidence_interval(
                n_sparsest[interval_size],
                precision_digits=precision_digits,
                search_region=child_limit_search_region)[1]
            # print("Result is %s" % child_limits[interval_size])
        self.last_limit_was_set_on_thisplusone_bins = np.argmin(child_limits)
        return child_limits[self.last_limit_was_set_on_thisplusone_bins]

    def generate_single_observation(self, n_trials):
        return self.distribution.rvs(size=n_trials)


class BinnedOptimumInterval_OneIntervalSize(TestStatistic):

    def __init__(self, parent, interval_size, *args, **kwargs):
        self.parent = parent
        self.interval_size = interval_size
        TestStatistic.__init__(self, *args, **kwargs)

    def get_values_and_likelihoods(self, mu, precision_digits=3):
        if self.use_pmf_cache and mu in self.pmf_cache:
            return self.pmf_cache[mu]

        if mu == 0:
            return np.array([0, 1]), np.array([1, 0])
        values, likelihoods = self.parent.get_values_and_likelihoods_for_interval_size(self.interval_size, mu)

        # Cache the pmf
        if self.use_pmf_cache:
            self.pmf_cache[mu] = values, likelihoods

        return values, likelihoods

    def __call__(self, *args):
        raise NotImplementedError("Please call through parent only!")

