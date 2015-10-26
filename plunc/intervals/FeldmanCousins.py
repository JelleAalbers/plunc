from .base import IntervalChoice
import numpy as np
from tqdm import tqdm


class FeldmanCousins(IntervalChoice):
    l_best_mu = None

    def get_single_interval_on_stat(self, *args, **kwargs):
        if self.l_best_mu is None:
            # Called for the first time
            # Compute the maximum-likelihood estimate of mu in the physical region (mu >0)
            # for each statistic value.
            # If the statistic is the number of elements, this is of course trivial!
            # TODO: this can be sped up by using the _trained_likelihood histogram, if available...
            self.l_best_mu = np.zeros(len(self.statistic.stats))
            self.best_mu = self.l_best_mu.copy()
            for i, s in enumerate(tqdm(self.statistic.stats, desc='Computing maximum likelihoods')):
                mus = self.statistic.mus.copy()
                ls = [self.statistic.likelihood(s, mu) for mu in mus]
                max_mu = mus[np.argmax(ls)]
                if max_mu < self.background:
                    max_mu = self.background
                self.best_mu[i] = max_mu
                self.l_best_mu[i] = self.statistic.likelihood(s, max_mu)

        return IntervalChoice.get_single_interval_on_stat(self, *args, **kwargs)

    def rank_stat_values(self, **kwargs):
        #         plt.plot(kwargs['likelihoods'], label='Likelihoods')
        #         plt.plot(self.l_under_max_mu, label='Likelihoods of max mu for this stat')
        #         plt.plot(kwargs['likelihoods']/self.l_under_max_mu, label='likelihood ratio')
        #         print(kwargs['likelihoods'].shape, self.l_under_max_mu.shape)
        return -kwargs['likelihoods']/self.l_best_mu