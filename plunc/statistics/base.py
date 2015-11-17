import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def round_to_digits(x, n_digits):
    """Rounds x to leading digits"""
    return round(x, n_digits - 1 - int(np.log10(x)) + (1 if x < 1 else 0))




class TestStatistic(object):
    """Base test statistic class
    """
    use_pmf_cache = False   # May fry your RAM, use wisely
    statistic_name = 's'
    n_trials = int(1e4)
    mu_dependent = False    # Set to True if the statistic is hypothesis-dependent

    def __init__(self, **kwargs):
        """Use kwargs to override any class-level settings"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.pmf_cache = {}

    def add_training_observations(self, training_observations, mu):
        """Train the likelihood function by passing Monte Carlo trials."""
        if self.training_done:
            raise ValueError("Training is already finished!")
        self._trained_likelihood.add([self(x, hypothesis=mu) for x in training_observations],
                                     [mu] * len(training_observations))

    def get_values_and_likelihoods(self, mu, desired_precision=None):
        # TODO: do something with desired_precision...
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
        """Return possible values, likelihoods. Can bin.
        By default uses a KDE (implemented as fine histogram + Gaussian filter)
        KDE bandwith = Silverman rule
        """
        # First take a very fine histogram, then Gauss filter
        # TODO: somehow make bins dependent on n_trials, and n_trials on desired precision...
        hist, bin_edges = np.histogram(values, bins=1000, density=True)
        bin_spacing = bin_edges[1] - bin_edges[0]
        bw = 1.06 * np.std(values) / len(values)**(1/5)
        hist = gaussian_filter1d(hist, sigma=bw / bin_spacing)
        return 0.5 * (bin_edges[1:] + bin_edges[:-1]), hist

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