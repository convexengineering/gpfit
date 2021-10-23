"""unit tests for fit module"""
import unittest
import pickle
import sys
from io import StringIO
from numpy import logspace, log10, log, vstack
from gpfit.fit import fit, MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestFit(unittest.TestCase):
    """Test fit class"""

    u = logspace(0, log10(3), 101)
    w = (u**2 + 3)/(u + 1)**2
    x = log(u)
    y = log(w)
    K = 3

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K, seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-2)
        self.assertEqual(f.__repr__(), (
            "w = 0.807159*(u_1)^-0.0703921\n"
            "w = 0.995106*(u_1)^-0.431386\n"
            "w = 0.92288*(u_1)^-0.247099"
        ))

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-4)
        self.assertEqual(f.__repr__(), (
            "w^3.44109 = 0.15339*(u_1)^0.584655\n"
            "    + 0.431128*(u_1)^-2.14831\n"
            "    + 0.415776*(u_1)^-2.14794"
        ))

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-5)
        self.assertEqual(f.__repr__(), (
            "1 = (0.947385/w^0.0920329)*(u_1)^0.0176859\n"
            "  + (0.992721/w^0.349639)*(u_1)^-0.201861\n"
            "  + (0.961596/w^0.116677)*(u_1)^-0.0112199"
        ))

    def test_fit_ma(self):
        f = fit(self.x, self.y, self.K, fit_type="ma", seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-2)
        self.assertEqual(f.__repr__(), (
            "w = 0.807159*(u_1)^-0.0703921\n"
            "w = 0.995106*(u_1)^-0.431386\n"
            "w = 0.92288*(u_1)^-0.247099"
        ))

    def test_fit_sma(self):
        f = fit(self.x, self.y, self.K, fit_type="sma", seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-4)
        self.assertEqual(f.__repr__(), (
            "w^3.44109 = 0.15339*(u_1)^0.584655\n"
            "    + 0.431128*(u_1)^-2.14831\n"
            "    + 0.415776*(u_1)^-2.14794"
        ))

    def test_fit_isma(self):
        f = fit(self.x, self.y, self.K, seed=SEED)
        self.assertTrue(f.errors["rms_rel"] < 1e-5)
        self.assertEqual(f.__repr__(), (
            "1 = (0.947385/w^0.0920329)*(u_1)^0.0176859\n"
            "  + (0.992721/w^0.349639)*(u_1)^-0.201861\n"
            "  + (0.961596/w^0.116677)*(u_1)^-0.0112199"
        ))

    def test_incorrect_inputs(self):
        with self.assertRaises(ValueError):
            MaxAffine(self.x, vstack((self.y, self.y)), self.K)

    def test_save_and_load(self):
        f1 = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        f1.save("fit.pkl")
        strings1 = f1.__repr__()
        with open("fit.pkl", "rb") as picklefile:
            f2 = pickle.load(picklefile)
        self.assertTrue(f2.errors["rms_rel"] < 1e-5)
        strings2 = f2.__repr__()
        self.assertEqual(strings1, strings2)

    def test_savetxt(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        f.savetxt("fit.txt")
        with open("fit.txt", "r", encoding="utf-8") as textfile:
            fitstring = textfile.read()
        self.assertEqual(fitstring, (
            "1 = (0.947385/w**0.0920329)*(u_1)**0.0176859\n"
            "  + (0.992721/w**0.349639)*(u_1)**-0.201861\n"
            "  + (0.961596/w**0.116677)*(u_1)**-0.0112199"
        ))

    def test_verbosity_1(self):
        captured_output = StringIO()
        sys.stdout = captured_output
        ImplicitSoftmaxAffine(self.x, self.y, self.K, verbosity=1, seed=SEED)
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

    def test_error(self):
        f1 = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        f2 = ImplicitSoftmaxAffine(self.x, 2*self.y, self.K, seed=SEED)
        self.assertAlmostEqual(f1.errors["rms_rel"], 8.12727e-07, places=11)
        self.assertAlmostEqual(f1.errors["rms_abs"], 7.43308e-07, places=11)
        self.assertAlmostEqual(f1.errors["rms_log"], 8.12726e-07, places=11)
        self.assertAlmostEqual(f1.errors["max_rel"], 3.52899e-06, places=10)
        self.assertAlmostEqual(f1.errors["max_abs"], 3.52899e-06, places=10)
        self.assertAlmostEqual(f2.errors["rms_rel"], 1.61329e-06, places=10)
        self.assertAlmostEqual(f2.errors["rms_abs"], 1.36908e-06, places=10)
        self.assertAlmostEqual(f2.errors["rms_log"], 1.61329e-06, places=10)
        self.assertAlmostEqual(f2.errors["max_rel"], 7.00807e-06, places=10)
        self.assertAlmostEqual(f2.errors["max_abs"], 7.00807e-06, places=10)
        self.assertEqual(f1.errors["rms_rel"], f1.error)


TESTS = [TestFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
