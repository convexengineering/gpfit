"""unit tests for fit module"""
import unittest
import numpy as np
from numpy import logspace, log10, log, vstack
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestFit(unittest.TestCase):
    """Test fit class"""

    np.random.seed(SEED)
    u = logspace(0, log10(3), 501)
    w = (u**2 + 3)/(u + 1)**2
    x = log(u)
    y = log(w)
    K = 3

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-2)

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-4)

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-5)

    #def test_incorrect_inputs(self):
    #    with self.assertRaises(ValueError):
    #        fit(self.x, vstack((self.y, self.y)), self.K, "MA")


TESTS = [TestFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
