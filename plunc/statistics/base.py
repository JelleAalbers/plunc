import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


class TestStatistic(object):
    """Base test statistic class
    """
    use_pmf_cache = True
    statistic_name = 's'
    n_trials = int(1e4)
    mu_dependent = False    # Set to True if the statistic is hypothesis-dependent

    def __init__(self, **kwargs):
        """Use kwargs to override any class-level settings"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.pmf_cache = {}

    def get_values_and_likelihoods(self, mu, precision_digits=3):
        # TODO: do something with precision_digits...
        if self.use_pmf_cache and mu in self.pmf_cache:
            return self.pmf_cache[mu]

        # Simulate self.n_trials observations
        # TODO: shortcut if self.__call__ is vectorized. Maybe implement in __call__ and have children override
        # something else instead (sounds nicer)
        values = np.zeros(self.n_trials)
        for i, obs in enumerate(self.observation_generator(mu, n_trials=self.n_trials)):
            values[i] = self(obs)

        # Summarize values to pmf
        # Allow statistic implementation to choose method for this, there is no universal solution
        # (e.g. very different for discrete or continuous statistics)
        # TODO: handle under-overflow here?
        values, likelihoods = self.build_pdf(values, mu=mu)
        likelihoods /= likelihoods.sum()

        # Cache the pmf
        if self.use_pmf_cache:
            self.pmf_cache[mu] = values, likelihoods

        return values, likelihoods

    def build_pdf(self, values, mu):
        """Return possible values, likelihoods. Can bin, can even use mu if desired (not usually needed).
        By default uses a KDE (implemented as fine histogram + Gaussian filter)
        KDE bandwith = Silverman rule
        """
        # First take a very fine histogram...
        # TODO: somehow make bins dependent on n_trials, and n_trials on desired precision...
        hist, bin_edges = np.histogram(values, bins=1000)
        hist = hist.astype(np.float)
        hist /= hist.sum()

        # ... then apply a Gaussian filter.
        # The filter is not applied on the outermost bins, since these might be accumulation points
        # we don't want to smear (e.g. a statistic generally may take an extreme value if there are no events)
        # The Bandwidth is determined by the Silverman rule of thumb, looking at the non-extreme values.
        # TODO: this behaviour should be configurable per-statistic, don't assume accumulation points by default
        bin_spacing = bin_edges[1] - bin_edges[0]
        non_extreme_values = values[(values != np.min(values)) & (values != np.max(values))]
        bandwidth = 1.06 * non_extreme_values.std() / len(non_extreme_values)**(1/5)
        center_hist = gaussian_filter1d(hist[1:-1], sigma=bandwidth / bin_spacing)
        # Ensure the Gauss filter has not changed the sum of the bins it was applied to:
        center_hist /= np.sum(hist[1:-1])/center_hist.sum()
        hist[1:-1] = center_hist
        if not np.isclose(np.sum(hist), 1):
            raise RuntimeError("WTF? Density histogram sums to %s, not 1 after filtering!" % np.sum(hist))

        # The values representing to the histogram bins are the bin centers...
        # ... except at the edges, where we use the outer boundaries.
        # This is again necessary to deal with accumulation points.
        values = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        values[0] = bin_edges[0]
        values[-1] = bin_edges[-1]
        return values, hist

    def probability(self, value, mu):
        """Returns probability of observing statistic = value under hypothesis mu
        This uses the likelihood trained from Monte Carlo: override this function
        if you have a better way of computing the likelihood
        """
        raise NotImplementedError

    def probability_leq(self, value, mu):
        """Returns probability of observing a statistic <= value under hypothesis mu
        This uses the likelihood trained from Monte Carlo: override this function
        if you have a better way of computing the likelihood
        """
        raise NotImplementedError

    def likelihood_of_observation(self, observation, mu):
        """Returns likelihood of observation under hypothesis mu"""
        raise NotImplementedError

    def __call__(self, observation, hypothesis=None):
        raise NotImplementedError

    def observation_generator(self, mu, n_trials=1):
        """Generate n_trials observations for the statistic under hypothesis mu"""
        n_per_trial = np.random.poisson(mu, n_trials)
        # Last array will always be empty, because we passed the very last index + 1 as split point
        return np.split(self.event_generator(np.sum(n_per_trial)),
                        np.cumsum(n_per_trial))[:-1]

    def event_generator(self, n):
        """Generate a single observation of n events"""
        return np.zeros(n)