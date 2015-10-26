import numpy as np
import multihist
from tqdm import tqdm


class TestStatistic(object):
    """Base test statistic class
    """
    statistic_name = 's'

    # Values to train
    mus = np.linspace(0, 100, 1000+1)
    stats = np.arange(0, 100)
    n_training_trials = 10000

    show_progress_bar = True
    auto_train = True
    mu_dependent = False    # Set to True if the statistic is hypothesis-dependent

    def __init__(self, **kwargs):
        """Use kwargs to override any class-level settings for:
            stats: possible values of the statistic to train
            mus: possible values of the hypothesis to train.
            n_training_trials
            auto_train
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        # We need one more bin edge than the values we want to store in it...
        stat_bins = np.concatenate((self.stats, [self.stats[-1] + 1e-6]))
        mu_bins = np.concatenate((self.mus, [self.mus[-1] + 1e-6]))
        self._trained_likelihood = multihist.Histdd(bins=(stat_bins, mu_bins),
                                                    axis_names=[self.statistic_name, 'mu'])
        self.training_done = False

        if self.auto_train:
            self.train_each_hypothesis(n_trials=self.n_training_trials)
            self.finish_training()

    def add_training_observations(self, training_observations, mu):
        """Train the likelihood function by passing Monte Carlo trials."""
        if self.training_done:
            raise ValueError("Training is already finished!")
        self._trained_likelihood.add([self(x, hypothesis=mu) for x in training_observations],
                                     [mu] * len(training_observations))

    def finish_training(self):
        if self.training_done:
            raise ValueError("Training is already finished!")
        self._trained_likelihood = self._trained_likelihood.normalize(axis=self.statistic_name)
        self.training_done = True

    def train_each_hypothesis(self, n_trials=1000, n_repetitions=1):
        for repetition_i in range(n_repetitions):
            # Shuffle is to make sure low and high mu are mixed and progress bar is accurate sooner :-)
            shuffled_mus = self.mus.copy()
            np.random.shuffle(shuffled_mus)
            if self.show_progress_bar:
                it = tqdm(shuffled_mus,
                           desc="Training likelihoods (%d/%d)" % (repetition_i + 1, n_repetitions))
            else:
                it = shuffled_mus
            for mu in it:
                self.add_training_observations(self.observation_generator(mu, n_trials),
                                               mu)

    def likelihood(self, value, mu):
        """Returns likelihood of the statistic value under hypothesis mu
        This uses the likelihood trained from Monte Carlo: override this function
        if you have a better way of computing the likelihood
        """
        if not self.training_done:
            raise ValueError("Training is not yet finished!")
        try:
            value[0]
        except (IndexError, TypeError):
            # Called with scalar for value
            try:
                return self._trained_likelihood[self._trained_likelihood.get_bin_indices((value, mu))]
            except IndexError:
                raise ValueError("value=%s, mu=%s is outside of the training range!" % (value, mu))
        else:
            # Called with arraylike for value
            return np.array([self.likelihood(x, mu) for x in value])

    def likelihood_leq(self, value, mu):
        """Returns likelihood of a statistic <= value under hypothesis mu
        This uses the likelihood trained from Monte Carlo: override this function
        if you have a better way of computing the likelihood
        """
        if not self.training_done:
            raise ValueError("Training is not yet finished!")
        try:
            value[0]
        except (IndexError, TypeError):
            # Called with scalar for value
            try:
                indices = self._trained_likelihood.get_bin_indices((value, mu))
                return np.sum(self._trained_likelihood[(slice(0, indices[0] + 1), indices[1])])
            except IndexError:
                raise ValueError("value=%s, mu=%s is outside of the training range!" % (value, mu))
        else:
            # Called with arraylike for value
            return np.array([self.likelihood_leq(x, mu) for x in value])

    def likelihood_of_observation(self, observation, mu):
        """Returns likelihood of observation under hypothesis mu"""
        return self.likelihood(self(observation, hypothesis=mu), mu)

    def __call__(self, observation, hypothesis=None):
        raise NotImplementedError

    def observation_generator(self, mu, n_trials=1):
        n_per_trial = np.random.poisson(mu, n_trials)
        # Last array will always be empty, because we passed the very last index + 1 as split point
        return np.split(self.event_generator(np.sum(n_per_trial)),
                        np.cumsum(n_per_trial))[:-1]

    def event_generator(self, n):
        return np.zeros(n)
        #return np.random.uniform(0, 1, np.sum(n_per_trial)