"unit tests for gpfit.print_fit module"
import unittest
import numpy as np
from gpfit.plot_fit import plot_fit_1d


class TestPlotFit(unittest.TestCase):
    "Unit tests for plot_fit_1d"
    N = 51
    u = np.logspace(0, np.log10(3), N)
    w = (u**2 + 3)/(u + 1)**2

    def test_plot_fit_1d(self):
        plot_fit_1d(self.u, self.w, K=2, fitclass='SMA', plotspace="linear")


TESTS = [TestPlotFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
