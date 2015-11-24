import numpy as np
import logging
from plunc.common import round_to_digits
from plunc.exceptions import SearchFailedException, InsufficientPrecisionError, OutsideDomainError

from plunc.WaryInterpolator import WaryInterpolator

class IntervalChoice(object):
    """Base interval choice method class
    """
    method = 'rank'    # 'rank' or 'threshold'
    threshold = float('inf')
    precision_digits = 2
    use_interval_cache = True
    wrap_interpolator = True
    background = 0
    confidence_level = 0.9
    max_hypothesis = 1e6
    interpolator_log_domain = (-1, 3)
    fixed_upper_limit = None
    fixed_lower_limit = None
    # Use only for testing:
    forbid_exact_computation = False

    def __init__(self, statistic, **kwargs):
        self.statistic = statistic
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.cl = self.confidence_level

        self.log = logging.getLogger(self.__class__.__name__)

        if self.wrap_interpolator:
            self.log.debug("Initializing interpolators")
            if self.fixed_lower_limit is None:
                self.low_limit_interpolator = WaryInterpolator(precision=10**(-self.precision_digits),
                                                               domain=self.interpolator_log_domain)
            if self.fixed_upper_limit is None:
                self.high_limit_interpolator = WaryInterpolator(precision=10**(-self.precision_digits),
                                                                domain=self.interpolator_log_domain)
            # "Joints" of the interpolator must have better precision than required of the interpolator results
            self.precision_digits += 1

        # Dictionary holding "horizontal" intervals: interval on statistic for each precision and hypothesis.
        self.cached_intervals = {}

    def get_interval_on_statistic(self, hypothesis, precision_digits):
        """Returns the self.cl confidence level interval on self.statistic for the event rate hypothesis
        The event rate here includes signal as well as identically distributed background.
        Intervals are inclusive = closed.
        """
        if self.use_interval_cache and (hypothesis, precision_digits) in self.cached_intervals:
            return self.cached_intervals[(hypothesis, precision_digits)]

        stat_values, likelihoods = self.statistic.get_values_and_likelihoods(hypothesis,
                                                                             precision_digits=precision_digits)
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
            self.cached_intervals[(hypothesis, precision_digits)] = low_lim, high_lim
        return low_lim, high_lim

    def get_confidence_interval(self, value, precision_digits, search_region, debug=False):
        """Performs the Neynman construction to get confidence interval on event rate (mu),
        if the statistic is observed to have value
        """
        log_value = np.log10(value)
        if self.wrap_interpolator:
            # Try to interpolate the limit from limits computed earlier
            self.log.debug("Trying to get values from interpolators")
            try:
                if self.fixed_lower_limit is None:
                    low_limit = 10**(self.low_limit_interpolator(log_value))
                else:
                    low_limit = self.fixed_lower_limit
                if self.fixed_upper_limit is None:
                    high_limit = 10**(self.high_limit_interpolator(log_value))
                else:
                    high_limit = self.fixed_upper_limit
                return low_limit, high_limit
            except InsufficientPrecisionError:
                self.log.debug("Insuffienct precision achieved by interpolators")
                if log_value > self.interpolator_log_domain[1]:
                    self.log.debug("Too high value to dare to start Neyman construction... raising exception")
                    # It is not safe to do the Neyman construction: too high statistics
                    raise
                self.log.debug("Log value %s is below interpolator log domain max %s "
                               "=> starting Neyman construction" % (log_value, self.interpolator_log_domain[1]))
            except OutsideDomainError:
                # The value is below the interpolator  domain (e.g. 0 while the domain ends at 10**0 = 1)
                pass

        if self.forbid_exact_computation:
            raise RuntimeError("Exact computation triggered")

        def is_value_in(mu):
            low_lim, high_lim = self.get_interval_on_statistic(mu + self.background,
                                                               precision_digits=precision_digits)
            return low_lim <= value <= high_lim

        # We first need one value in the interval to bound the limit searches
        try:
            true_point, low_search_bound, high_search_bound = search_true_instance(is_value_in,
                                                                                   *search_region,
                                                                                   precision_digits=precision_digits)
        except SearchFailedException as e:
            self.log.debug("Exploratory search could not find a single value in the interval! "
                           "This is probably a problem with search region, or simply a very extreme case."
                           "Original exception: %s" % str(e))
            if is_value_in(0):
                self.log.debug("Oh, ok, only zero is in the interval... Returning (0, 0)")
                return 0, 0
            return 0, float('inf')

        self.log.debug(">>> Exploratory search completed: %s is in interval, "
                       "search for boundaries in [%s, %s]" % (true_point, low_search_bound, high_search_bound))

        if self.fixed_lower_limit is not None:
            low_limit = self.fixed_lower_limit
        elif is_value_in(low_search_bound):
            # If mu=0 can't be excluded, we're apparently only setting an upper limit (mu <= ..)
            low_limit = 0
        else:
            low_limit = bisect_search(is_value_in, low_search_bound, true_point, precision_digits=precision_digits)
        self.log.debug(">>> Low limit found at %s" % low_limit)

        if self.fixed_upper_limit is not None:
            low_limit = self.fixed_upper_limit
        elif is_value_in(high_search_bound):
            # If max_mu can't be excluded, we're apparently only setting a lower limit (mu >= ..)
            high_limit = float('inf')
        else:
            high_limit = bisect_search(is_value_in, true_point, high_search_bound, precision_digits=precision_digits)
        self.log.debug(">>> High limit found at %s" % high_limit)

        if self.wrap_interpolator:
            # Add the values to the interpolator, if they are within the domain
            # TODO: Think about dealing with inf
            if self.interpolator_log_domain[0] <= log_value <= self.interpolator_log_domain[1]:
                if self.fixed_lower_limit is None:
                    self.low_limit_interpolator.add_point(log_value, np.log10(low_limit))
                if self.fixed_upper_limit is None:
                    self.high_limit_interpolator.add_point(log_value, np.log10(high_limit))

        return low_limit, high_limit

    def score_stat_values(self, **kwargs):
        # Return "rank" of each hypothesis. Hypotheses with highest ranks will be included first.
        raise NotImplementedError()

    def __call__(self, observation, precision_digits=None, search_region=None):
        """Perform Neynman construction to get confidence interval on event rate for observation.
        """
        if precision_digits is None:
            precision_digits = self.precision_digits
        if search_region is None:
            search_region = [0, round_to_digits(10 + 3 * len(observation), precision_digits)]
        if self.statistic.mu_dependent:
            value = self.statistic(observation, self.statistic.mus)
        else:
            value = self.statistic(observation, None)
        self.log.debug("Statistic evaluates to %s" % value)
        return self.get_confidence_interval(value, precision_digits=precision_digits, search_region=search_region)


def search_true_instance(f, a, b, precision_digits=3, maxiter=10, log=None):
    """Find x in [a, b] where f is True, limiting search to values with precision_digits significant figures.
    Returns x, low_bound, high_bound where low_bound and high_bound are either the search bounds a or b, or closer
    values to x where f was still found to be False.
    # TODO: If asked for precision_digits=5, first search with precision_digits=1, then 2, etc.

    print(search_true_instance(lambda x: 11 < x < 13, 0, 40))
    print(search_true_instance(lambda x: x < 13, 0, 1000))
    """
    log = logging.getLogger('search_true_instance')

    values_searched = [a, b]
    log.debug("Starting exploratory search in [%s, %s]" % (a, b))

    for iter_i in range(maxiter):
        # First test halfway, point then 1/4 and 3/4, then 1/8, 3/8, 5/8, 7/8, etc.
        fractions = 2**(iter_i + 1)
        search_points = [round_to_digits(a + (b - a)*fr, precision_digits)
                         for fr in np.arange(1, fractions, 2)/fractions]
        log.debug("Searching %s - %s (%d points)" % (search_points[0], search_points[-1], len(search_points)))

        for x_i, x in enumerate(search_points):
            if f(x):
                values_searched = np.array(values_searched)
                return x, np.max(values_searched[values_searched < x]), np.min(values_searched[values_searched > x])
            else:
                values_searched.append(x)

        if len(search_points) > 1 and np.any(np.diff(search_points) == 0):
            raise SearchFailedException("No true value found in search region [%s, %s], "
                                        "but search depth now lower than precision digits (%s). "
                                        "Iteration %d." % (a, b, precision_digits, iter_i))

    raise ValueError("Exploratory search failed to converge or terminate - bug? excessive precision?")


def bisect_search(f, a, b, precision_digits=2, maxiter=1e2):
    """Find x in [a, b] where f changes from True to False by bisection,
    limiting search to values with precision_digits significant figures.
    This is useful if f can cache its results: otherwise just use e.g. scipy.optimize.brentq with rtol.
    Avoid scipy.optimize.bisect with rtol, results seem seem to depend heavily on initial bounds: bug??
    # TODO: If asked for precision_digits=5, first search with precision_digits=1, then 2, etc.
    """
    log = logging.getLogger('bisect_search')

    # Which of the bounds gives True? Can't be both!
    if f(a) == f(b):
        raise ValueError("f must not be true or false on both bounds")
    true_on_a = f(a)

    log.debug("Starting search between %s (%s) and %s (%s)"
              " with %d precision digits" % (a, f(a), b, f(b), precision_digits))

    # Do a bisection search, sticking to precision_digits
    for iter_i in range(int(maxiter)):

        # Find the new bisection point
        x = (a + b) / 2
        x = round_to_digits(x, precision_digits)

        # If we are down to a single point, return that
        if x == a or x == b:
            return x
        true_on_x = f(x)

        # Update the appropriate bound
        if true_on_a:
            if true_on_x:
                a = x
            else:
                b = x
        else:
            if true_on_x:
                b = x
            else:
                a = x

        log.debug("Iteration %d, searching between [%s and %s], last x was %s (%s)" % (iter_i, a, b, x, true_on_x))

    else:
        raise RuntimeError("Infinite loop encountered in bisection search!")

# Earlier custom minimizer idea
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
