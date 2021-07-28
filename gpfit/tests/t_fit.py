"""unit tests for fit module"""
import unittest
import pickle
import sys
from io import StringIO
import numpy as np
from numpy import logspace, log10, log, vstack
from gpfit.fit import fit, MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestFit(unittest.TestCase):
    """Test fit class"""

    np.random.seed(SEED)
    u = logspace(0, log10(3), 101)
    w = (u**2 + 3)/(u + 1)**2
    x = log(u)
    y = log(w)
    K = 3

    def test_max_affine(self):
        np.random.seed(SEED)
        f = MaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-2)
        self.assertEqual(f.__repr__(), (
            "w = 0.807159*(u_1)^-0.0703921\n"
            "w = 0.995106*(u_1)^-0.431386\n"
            "w = 0.92288*(u_1)^-0.247099"
        ))

    def test_softmax_affine(self):
        np.random.seed(SEED)
        f = SoftmaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-4)
        self.assertEqual(f.__repr__(), (
            "w^3.44109 = 0.15339*(u_1)^0.584655\n"
            "    + 0.431128*(u_1)^-2.14831\n"
            "    + 0.415776*(u_1)^-2.14794"
        ))

    def test_implicit_softmax_affine(self):
        np.random.seed(SEED)
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-5)
        self.assertEqual(f.__repr__(), (
            "1 = (0.947385/w^0.0920329)*(u_1)^0.0176859\n"
            "  + (0.992721/w^0.349639)*(u_1)^-0.201861\n"
            "  + (0.961596/w^0.116677)*(u_1)^-0.0112199"
        ))

    def test_fit_ma(self):
        np.random.seed(SEED)
        f = fit(self.x, self.y, self.K, fit_type="ma")
        self.assertTrue(f.error["rms"] < 1e-2)
        self.assertEqual(f.__repr__(), (
            "w = 0.807159*(u_1)^-0.0703921\n"
            "w = 0.995106*(u_1)^-0.431386\n"
            "w = 0.92288*(u_1)^-0.247099"
        ))

    def test_fit_sma(self):
        np.random.seed(SEED)
        f = fit(self.x, self.y, self.K, fit_type="sma")
        self.assertTrue(f.error["rms"] < 1e-4)
        self.assertEqual(f.__repr__(), (
            "w^3.44109 = 0.15339*(u_1)^0.584655\n"
            "    + 0.431128*(u_1)^-2.14831\n"
            "    + 0.415776*(u_1)^-2.14794"
        ))

    def test_fit_isma(self):
        np.random.seed(SEED)
        f = fit(self.x, self.y, self.K)
        self.assertTrue(f.error["rms"] < 1e-5)
        self.assertEqual(f.__repr__(), (
            "1 = (0.947385/w^0.0920329)*(u_1)^0.0176859\n"
            "  + (0.992721/w^0.349639)*(u_1)^-0.201861\n"
            "  + (0.961596/w^0.116677)*(u_1)^-0.0112199"
        ))

    def test_incorrect_inputs(self):
        with self.assertRaises(ValueError):
            MaxAffine(self.x, vstack((self.y, self.y)), self.K)

    def test_save_and_load(self):
        np.random.seed(SEED)
        f1 = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        f1.save("artifacts/fit.pkl")
        strings1 = f1.__repr__()
        f2 = pickle.load(open("artifacts/fit.pkl", "rb"))
        self.assertTrue(f2.error["rms"] < 1e-5)
        strings2 = f2.__repr__()
        self.assertEqual(strings1, strings2)

    def test_savetxt(self):
        np.random.seed(SEED)
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        f.savetxt("artifacts/fit.txt")
        with open("artifacts/fit.txt", "r") as f:
            fitstring = f.read()
        self.assertEqual(fitstring, (
            "1 = (0.947385/w**0.0920329)*(u_1)**0.0176859\n"
            "  + (0.992721/w**0.349639)*(u_1)**-0.201861\n"
            "  + (0.961596/w**0.116677)*(u_1)**-0.0112199"
        ))

    def test_verbosity_1(self):
        captured_output = StringIO()
        sys.stdout = captured_output
        np.random.seed(SEED)
        ImplicitSoftmaxAffine(self.x, self.y, self.K, verbosity=1)
        sys.stdout = sys.__stdout__
        expected_output = (
            "Generated ImplicitSoftmaxAffine fit with 3 terms.\n"
            "\n"
            "Fit\n"
            "---\n"
            "1 = (0.947385/w^0.0920329)*(u_1)^0.0176859\n"
            "  + (0.992721/w^0.349639)*(u_1)^-0.201861\n"
            "  + (0.961596/w^0.116677)*(u_1)^-0.0112199\n"
            "\n"
            "Error\n"
            "-----\n"
            "RMS: 8.1e-05%\n"
            "Max: 0.00035%\n\n"
        )
        self.assertEqual(expected_output, captured_output.getvalue())


TESTS = [TestFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
