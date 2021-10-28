"unit tests for get_initial_parameters function"
import unittest
from numpy import arange, newaxis, vstack, log, exp
from numpy.random import random_sample
from gpfit.maths.initialize import get_initial_parameters


class TestMaxAffineInitK2(unittest.TestCase):
    """
    This unit test ensures that max affine init produces an array
    of the expected shape and size
    """

    x = arange(0.0, 16.0)[:, newaxis]
    y = arange(0.0, 16.0)[:, newaxis]
    K = 2
    ba = get_initial_parameters(x, y, K)

    def test_ba_ndim_k2(self):
        self.assertEqual(self.ba.ndim, 2)

    def test_ba_shape_k2(self):

        _, dimx = self.x.shape
        self.assertEqual(self.ba.shape, (dimx + 1, self.K))


class TestMaxAffineInitK4(unittest.TestCase):
    """
    This unit test ensures that max affine init produces an array
    of the expected shape and size
    """

    Vdd = random_sample(1000) + 1
    Vth = 0.2*random_sample(1000) + 0.2
    P = Vdd**2 + 30*Vdd*exp(-(Vth - 0.06*Vdd)/0.039)
    u = vstack((Vdd, Vth))
    x = log(u)
    y = log(P)
    x = x.T
    y = y.reshape(y.size, 1)
    K = 4

    ba = get_initial_parameters(x, y, K)

    def test_ba_shape_k4(self):
        self.assertEqual(self.ba.shape, (3, 4))


TESTS = [TestMaxAffineInitK2, TestMaxAffineInitK4]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
