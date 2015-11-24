import unittest
import numpy as np

import logging
import sys

from plunc.intervals.basic import CentralCI
from plunc.statistics import NumberOfEvents
from plunc.exceptions import InsufficientPrecisionError

# 90% confidence level on Poisson rate if n=20 or n=120 events observed.
# Values taken from calculator at http://statpages.org/confint.html
POISSON_90_CI_20 = (13.2547, 29.0620)
POISSON_90_CI_120 = (102.5677, 139.6439)


class TestIntervalChoice(unittest.TestCase):

    def test_ci(self):
        """Compute a single Poisson confidence interval"""
        poisson_ci = CentralCI(statistic=NumberOfEvents(), confidence_level=0.9, precision_digits=5)
        ll, hl = poisson_ci(np.ones(20))

        # Note assertAlmostEqual's 'places' refers to digits past the decimal point
        self.assertAlmostEqual(ll, POISSON_90_CI_20[0], places=3)
        self.assertAlmostEqual(hl, POISSON_90_CI_20[1], places=3)

    def test_wrap_interpolator(self):
        poisson_ci = CentralCI(statistic=NumberOfEvents(), confidence_level=0.9, precision_digits=2,
                               wrap_interpolator=True)

        # Precompute some values
        logging.getLogger().setLevel(logging.INFO)
        for n in np.arange(100):
            poisson_ci(np.ones(n))
        logging.getLogger().setLevel(logging.DEBUG)

        poisson_ci.forbid_exact_computation = True

        # Test the interpolator at one of the joints
        ll, hl = poisson_ci(np.ones(20))
        self.assertAlmostEqual(np.log10(ll),
                               np.log10(POISSON_90_CI_20[0]), places=1)
        self.assertAlmostEqual(np.log10(hl),
                               np.log10(POISSON_90_CI_20[1]), places=1)

        # Test the extrapolation
        ll, hl = poisson_ci(np.ones(120))
        self.assertAlmostEqual(np.log10(ll), np.log10(POISSON_90_CI_120[0]), places=1)
        self.assertAlmostEqual(np.log10(hl), np.log10(POISSON_90_CI_120[1]), places=1)

        # Test a ridiculously far extrapolation: should raise error
        self.assertRaises(InsufficientPrecisionError, poisson_ci, np.ones(int(1e4)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stderr)
    unittest.main()
