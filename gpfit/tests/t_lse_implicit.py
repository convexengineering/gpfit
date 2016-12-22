"unit tests for implicit log-sum-exp implementation"
import unittest
import numpy as np
from numpy import array, arange
from gpfit.lse_implicit import lse_implicit

SEED = 33404

class TestLSEimplicit1D(unittest.TestCase):
    "tests with one-dimensional input"

    x = arange(1., 31.).reshape(15, 2)
    alpha = arange(1., 3.)
    y, dydx, dydalpha = lse_implicit(x,alpha)

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

    np.random.seed(SEED)

    K = 4
    x = np.random.rand(1000,K)
    alpha = array([1.499, 13.703, 3.219, 4.148])

    y, dydx, dydalpha = lse_implicit(x, alpha)

    def test_dydx_shape(self):
        self.assertEqual(self.dydx.shape, self.x.shape)

    def test_dydalpha_shape(self):
        self.assertEqual(self.dydalpha.shape, self.x.shape)

tests = [TestLSEimplicit1D,
         TestLSEimplicit2D]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
