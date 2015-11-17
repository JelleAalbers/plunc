import numpy as np
# Tried bisect, but results after rtol stop seem to depend heavily on initial bounds: bug??
from scipy.optimize import brentq

class IntervalChoice(object):
    """Base interval choice method class
    """
    method = 'rank'    # 'rank' or 'threshold'
    threshold = float('inf')
    desired_precision = 0.01
    use_interval_cache = False      # May fry your RAM, use wisely
    max_hypothesis = 1e6

    def __init__(self, statistic, confidence_level=0.9, background=0):
        self.cl = confidence_level
        self.statistic = statistic
        self.background = background

        # Dictionary holding "horizontal" intervals: interval on statistic for each precision and hypothesis.
        self.cached_intervals = {}

    def get_interval_on_statistic(self, hypothesis, desired_precision):
        """Returns the self.cl confidence level interval on self.statistic for the event rate hypothesis
        The event rate here includes signal as well as identically distributed background.
        Intervals are inclusive = closed.
        """
        if self.use_interval_cache and (hypothesis, desired_precision) in self.cached_intervals:
            return self.cached_intervals[(hypothesis, desired_precision)]

        # Remember hypothesis here includes background!
        stat_values, likelihoods = self.statistic.get_values_and_likelihoods(hypothesis,
                                                                             desired_precision=desired_precision)
        likelihoods = likelihoods / np.sum(likelihoods)

        # Score each statistic value (method-dependent)
        stat_value_scores = self.score_stat_values(statistic_values=stat_values,
                                                   likelihoods=likelihoods,
                                                   hypothesis=hypothesis)
        if self.method == 'threshold':
            # Include all intervals that score higher than some threshold
            values_in_interval = stat_values[stat_value_scores > self.get_threshold()]

        else:
            # Include the values with highest score first, until we reach the desired confidence level
            # TODO: wouldn't HIGHEST score first be more user-friendly?
            ranks = np.argsort(stat_value_scores)
            train_values_sorted = stat_values[ranks]
            likelihoods_sorted = likelihoods[ranks]

            # Find the last value to include
            # (= first value that takes the included probability over the required confidence level)
            sum_lhoods = np.cumsum(likelihoods_sorted)
            # import matplotlib.pyplot as plt
            # plt.plot(sum_lhoods, marker='*')
            # for x, y, ann in zip(np.arange(len(sum_lhoods)), sum_lhoods, train_values_sorted):
            #     plt.annotate(str(ann), xy=(x,y))
            # plt.show()
            last_index = np.where(sum_lhoods > self.cl)[0][0]   # TODO: can fail?
            values_in_interval = train_values_sorted[:last_index + 1]

        low_lim, high_lim = values_in_interval.min(), values_in_interval.max()

        # If we included all values given up until a boundary, don't set that boundary as a limit
        if low_lim == np.min(stat_values):
            low_lim = 0
        if high_lim == np.max(stat_values):
            high_lim = float('inf')

        # Check for non-continuity
        # if len(values_in_interval) != np.where(train_values == high_lim)[0][0] - np.where(train_values == low_lim)[0][0] + 1:
        #     raise ValueError("Values in interval %s for hypothesis %s "
        #                      "are not continuous!" % (sorted(values_in_interval), hypothesis))

        # Cache and return upper and lower limit on the statistic
        if self.use_interval_cache:
            self.cached_intervals[(hypothesis, desired_precision)] = low_lim, high_lim
        return low_lim, high_lim

    def is_value_included(self, value, hypothesis, desired_precision):
        low_lim, high_lim = self.get_interval_on_statistic(hypothesis, desired_precision)
        return low_lim <= value <= high_lim

    def get_confidence_interval(self, value, desired_precision=None, max_mu=None, guess=None):
        """Perform Neynman construction to get confidence interval on event rate,
        if the statistic is observed to have value
        """
        if max_mu is None:
            max_mu = self.max_hypothesis
        if desired_precision is None:
            desired_precision = self.desired_precision

        # We first need one value in the interval to bound the limit searches
        # TODO: Search this in some way... How?
        if not self.is_value_included(value, guess, desired_precision):
            raise ValueError("Guess must be in the interval; you gave %s, which is not." % guess)

        if self.is_value_included(value, 0, desired_precision):
            # If mu=0 can't be excluded, we're apparently only setting an upper limit (mu <= ..)
            low_limit = 0
        else:
            low_limit = brentq(lambda mu: -1 + mu - max_mu if self.is_value_included(value, mu,
                                                                                     desired_precision) else 1 + mu,
                               0, guess, rtol=desired_precision)

        if self.is_value_included(value, max_mu, desired_precision):
            # If max_mu can't be excluded, we're apparently only setting a lower limit (mu >= ..)
            high_limit = float('inf')
        else:
            high_limit = brentq(lambda mu: 1 + mu if self.is_value_included(value, mu,
                                                                            desired_precision) else -1 + mu - max_mu,
                                guess, max_mu, rtol=desired_precision)

        return low_limit, high_limit

    def score_stat_values(self, **kwargs):
        # Return "rank" of each hypothesis. Hypotheses with highest ranks will be included first.
        raise NotImplementedError()

    def __call__(self, observation, desired_precision=0.01, guess=None):
        """Perform Neynman construction to get confidence interval on event rate for observation.
        """
        if guess is None:
            guess = len(observation)
        if desired_precision is None:
            desired_precision = self.desired_precision
        if self.statistic.mu_dependent:
            value = self.statistic(observation, self.statistic.mus)
        else:
            value = self.statistic(observation, None)
        return self.get_confidence_interval(value, desired_precision=desired_precision, guess=guess)


# Mu is not necessarily integer. But if it was...
# def bisect_search(f, a, b, rtol, maxiter=1e6):
#     """Zoom in to integer x where f changes from True to False between integers a and b.
#     If (a - b) < rtol or b = a + 1, stops and returns last integer for which f is True.
#     """
#     # Which of the bounds gives True? Can't be both!
#     assert f(a) != f(b)
#     if f(a):
#         last_true = a
#         assert not f(b)
#     else:
#         assert f(b)
#         last_true = b
#
#     # Do the bisection search
#     for _ in range(maxiter):
#         if b == a + 1:
#             return last_true
#         x = (a + b) // 2
#         if (a - b)/x < rtol:
#             return last_true
#         c = (a + b)/2
#
#     else:
#         raise RuntimeError("Infinite loop encountered in bisection search!")

# Custom limit setter = bad idea
# class LimitTracker(object):
#
#     def __init__(self, interval_choice, value, max_hypothesis=1e6, desired_precision=0.01):
#         self.ivc = interval_choice
#         self.value = value
#
#         # Keep track of the lower & upper bound on the low and high limit
#         self.low_limit_bounds = [0, max_hypothesis]
#         self.high_limit_bounds = [0, max_hypothesis]
#
#         self.has_low_limit = True
#         self.has_high_limit = True
#
#     @property
#     def is_twosided(self):
#         return self.has_high_limit and self.has_low_limit
#
#     def try_hypothesis(self, hypothesis):
#         if self.ivc.is_value_included(self.value, hypothesis):
#             if hypothesis < self.low_limit_bounds[1]:
#                 # Constrain the high bound of the low limit
#                 self.low_limit_bounds[1] = hypothesis
#             else:
#                 # Constrain the low bound of the high limit
#                 assert hypothesis > self.high_limit_bounds[0]
#                 self.high_limit_bounds[0] = hypothesis
#         else:
#             if hypothesis > self.high_limit_bounds[0]:
#                 # Constrain the high bound of the high limit
#                 assert hypothesis < self.high_limit_bounds[1]
#                 self.high_limit_bounds[1] = hypothesis
#             else:
#                 # Constrain the low bound of the low limit
#                 assert hypothesis > self.low_limit_bounds[0]
#                 self.low_limit_bounds[0] = hypothesis
#         assert self.high_limit_bounds[1] >= self.high_limit_bounds[0]
#         assert self.low_limit_bounds[1] >= self.low_limit_bounds[0]
#         if self.is_twosided:
#             assert self.high_limit_bounds[0] >= self.low_limit_bounds[0]
#
#     def improve_limit(self):
#         # Which limit should we work on?
#         if self.is_twosided:
#             # Work on the limit which currently has the widest bounds:
#             if self.low_limit_bounds[1] - self.low_limit_bounds[0] > \
#                self.high_limit_bounds[1] - self.high_limit_bounds[0]:
#                 work_on_high_limit = False
#         else:
#             work_on_high_limit = not self.no_high_limit
#
#         if work_on_high_limit:
#             self.try_hypothesis(0.5 * (self.high_limit_bounds[0] + self.high_limit_bounds[1]))
