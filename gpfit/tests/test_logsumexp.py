"unit tests for log-sum-exp functions"
import unittest
import numpy as np
from numpy import array, arange
from gpfit.maths.logsumexp import lse_implicit, lse_scaled


class TestLSEimplicit1D(unittest.TestCase):
    "tests with one-dimensional input"

    x = arange(1.0, 31.0).reshape(15, 2)
    alpha = arange(1.0, 3.0)
    y, dydx, dydalpha = lse_implicit(x, alpha)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.shape[0])

    def test_dydx_ndim(self):
        self.assertEqual(self.dydx.ndim, 2)

    def test_dydx_shape_0(self):
        self.assertEqual(self.dydx.shape[0], self.x.shape[0])

    def test_dydalpha_ndim(self):
        self.assertEqual(self.dydalpha.ndim, 2)

    def test_dydalpha_size(self):
        self.assertEqual(self.dydalpha.shape[0], self.x.shape[0])


class TestLSEimplicit2D(unittest.TestCase):
    "tests with 2D input"

    K = 4
    x = np.random.rand(1000, K)
    alpha = array([1.499, 13.703, 3.219, 4.148])

    y, dydx, dydalpha = lse_implicit(x, alpha)

    def test_dydx_shape(self):
        self.assertEqual(self.dydx.shape, self.x.shape)

    def test_dydalpha_shape(self):
        self.assertEqual(self.dydalpha.shape, self.x.shape)


class TestLSEScaled(unittest.TestCase):
    "Test lse_implicit"

    x = arange(1.0, 31.0).reshape(15, 2)
    alpha = 1.0
    y, dydx, dydalpha = lse_scaled(x, alpha)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.shape[0])

    def test_dydx_ndim(self):
        self.assertEqual(self.dydx.ndim, 2)

    def test_dydx_shape_0(self):
        self.assertEqual(self.dydx.shape[0], self.x.shape[0])

    def test_dydalpha_ndim(self):
        self.assertEqual(self.dydalpha.ndim, 1)

    def test_dydalpha_size(self):
        self.assertEqual(self.dydalpha.size, self.x.shape[0])

    # test alpha is integer? negative? 0? array?


TESTS = [
    TestLSEimplicit1D,
    TestLSEimplicit2D,
    TestLSEScaled,
]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
