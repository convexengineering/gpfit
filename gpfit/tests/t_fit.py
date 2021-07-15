"""unit tests for fit module"""
import unittest
from numpy import logspace, log10, log, vstack
from gpfit.fit import fit


class TestFit(unittest.TestCase):
    """Test fit function"""

    u = logspace(0, log10(3), 501)
    w = (u**2 + 3)/(u + 1)**2
    x = log(u)
    y = log(w)
    K = 3

    def test_rms_error(self):
        _, rms_error = fit(self.x, self.y, self.K, "SMA")
        self.assertTrue(rms_error < 1e-4)
        _, rms_error = fit(self.x, self.y, self.K, "ISMA")
        self.assertTrue(rms_error < 1e-5)
        _, rms_error = fit(self.x, self.y, self.K, "MA")
        self.assertTrue(rms_error < 1e-2)

    def test_incorrect_inputs(self):
        with self.assertRaises(ValueError):
            fit(self.x, vstack((self.y, self.y)), self.K, "MA")


TESTS = [TestFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
