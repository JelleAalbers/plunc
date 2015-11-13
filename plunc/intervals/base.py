import numpy as np
from tqdm import tqdm


class IntervalChoice(object):
    """Base interval choice method class
    """
    method = 'rank'    # 'rank' or 'threshold'
    threshold = float('inf')

    def __init__(self, statistic, confidence_level=0.9, background=0):
        self.cl = confidence_level
        self.statistic = statistic
        self.background = background

        # Train the intervals on each hypothesis
        self.lowlims_stat = np.zeros(len(self.statistic.mus))
        self.highlims_stat = np.zeros(len(self.statistic.mus))
        for i, hyp in enumerate(tqdm(self.statistic.mus, desc='Getting stat intervals for each mu')):
            self.lowlims_stat[i], self.highlims_stat[i] = self.get_single_interval_on_stat(hyp + self.background)

    def get_single_interval_on_stat(self, hypothesis):
        if hypothesis > self.statistic.mus[-1]:
            # Hypothesis is outside the training range
            # Return interval for last hypothesis in training range
            return self.get_single_interval_on_stat(self.statistic.mus[-1])

        # Remember hypothesis here includes background!
        train_values = self.statistic.stats
        likelihoods = self.statistic.likelihood(train_values, hypothesis)
        likelihoods = likelihoods / np.sum(likelihoods)

        # Sort train_values and likelihoods by decreasing priority of inclusion
        values_for_ranking = self.rank_stat_values(statistic_values=train_values,
                                                   likelihoods=likelihoods,
                                                   hypothesis=hypothesis)
        if self.method == 'threshold':
            values_in_interval = train_values[values_for_ranking > self.get_threshold()]

        else:
            # Include the values with highest rank first until we reach the desired confidence level
            ranks = np.argsort(values_for_ranking)  # TODO: ascending or descending??
            train_values_sorted = train_values[ranks]
            likelihoods_sorted = likelihoods[ranks]

            # Find the last value to include
            sum_lhoods = np.cumsum(likelihoods_sorted)
            #             plt.plot(sum_lhoods, marker='*')
            #             for x, y, ann in zip(np.arange(len(sum_lhoods)), sum_lhoods, train_values_sorted):
            #                 plt.annotate(str(ann), xy=(x,y))
            last_index = np.where(sum_lhoods > self.cl)[0][0]   # TODO: can fail?
            values_in_interval = train_values_sorted[:last_index + 1]

        # Check for about non-continuity
        low_lim, high_lim = values_in_interval.min(), values_in_interval.max()
        if len(values_in_interval) != np.where(train_values == high_lim)[0][0] - np.where(train_values == low_lim)[0][0] + 1:
            raise ValueError("Values in interval %s for hypothesis %s "
                             "are not continuous!" % (sorted(values_in_interval), hypothesis))

        # Return upper and lower limit on the statistic
        return low_lim, high_lim

    def get_confidence_interval(self, value):
        is_in = (self.lowlims_stat <= value) & (value <= self.highlims_stat)
        # Find upper and lower limit on hypothesis
        # TODO: use interpolating functions to give more precise limit
        indices_in = np.where(is_in)[0]
        if not len(indices_in):
            # Statistic returns empty limit
            return (0, 0)
        # Assignment to lowlim etc will remain safe: because we use advanced indexing, we get a view
        lowlim, highlim = self.statistic.mus[[indices_in[0], indices_in[-1]]]
        if highlim == self.statistic.mus[-1]:
            highlim = float('inf')
        return lowlim, highlim

    def rank_stat_values(hypotheses, likelihoods):
        # Return "rank" of each hypothesis. Hypotheses with highest ranks will be included first.
        raise NotImplementedError()

    def __call__(self, observation):
        if self.statistic.mu_dependent:
            value = self.statistic(observation, self.statistic.mus)
        else:
            value = self.statistic(observation, None)
        return self.get_confidence_interval(value)