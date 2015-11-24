import unittest
import numpy as np
import logging
import sys


from plunc.WaryInterpolator import WaryInterpolator, OutsideDomainError
from plunc.exceptions import InsufficientPrecisionError, OutsideDomainError


class TestWaryInterpolator(unittest.TestCase):

    def test_add_points(self):
        itp1 = WaryInterpolator(domain=[0, 5])
        itp1.add_points([0, 4, 2], [10, 30, 20])

        itp2 = WaryInterpolator(points=(0, 2, 4), values=(10, 20, 30), domain=[0, 5])

        np.testing.assert_array_equal(itp1.points, itp2.points)
        np.testing.assert_array_equal(itp1.values, itp2.values)

        itp1.add_point(4, 40)
        itp2.add_point(4, 40)

        np.testing.assert_array_equal(itp1.points, itp2.points)
        np.testing.assert_array_equal(itp1.values, itp2.values)

    def test_interpolate(self):
        itp = WaryInterpolator(if_higher='extrapolate', if_lower='raise', domain=(0, 5))

        self.assertRaises(InsufficientPrecisionError, itp, 0)

        itp.add_points([0, 2, 4], [10, 20, 30])
        self.assertEqual(itp(0), 10)
        self.assertAlmostEqual(itp(1), 15)
        self.assertAlmostEqual(itp(5), 35)
        self.assertRaises(OutsideDomainError, itp, -5)

    def test_precision(self):
        itp_x2 = WaryInterpolator(points=(0, 1, 2), values=(0, 1, 4), precision=0.01, domain=(0, 5))
        self.assertRaises(InsufficientPrecisionError, itp_x2, 0.5)

        xs = np.array([0.485, 0.495, 0.505, 0.515])
        itp_x2.add_points(xs, xs**2)

        self.assertAlmostEqual(itp_x2(0.5), (0.495**2 + 0.505**2)/2)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    unittest.main()