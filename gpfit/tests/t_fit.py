"""unit tests for fit module"""
import unittest
import pickle
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
        np.random.seed(SEED)
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-5)
        strings = f.print_fit()

    def test_incorrect_inputs(self):
        with self.assertRaises(ValueError):
            MaxAffine(self.x, vstack((self.y, self.y)), self.K)

    def test_save_and_load(self):
        np.random.seed(SEED)
        f1 = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        f1.save("artifacts/fit.pkl")
        strings1 = f1.print_fit()
        f2 = pickle.load(open("artifacts/fit.pkl", "rb"))
        self.assertTrue(f2.error["rms"] < 1e-5)
        strings2 = f2.print_fit()
        self.assertEqual(strings1, strings2)

TESTS = [TestFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
