"unit tests for gpfit.print_fit module"
import unittest
import numpy as np
from gpfit.plot_fit import plot_fit_1d


class TestPlotFit(unittest.TestCase):
    "Unit tests for plot_fit_1d"

    def test_plot_fit_1d(self):
        N = 51
        U = np.logspace(0, np.log10(3), N)
        W = (U**2+3)/(U+1)**2
        plot_fit_1d(U, W, K=2, fitclass='SMA', plotspace="linear")


TESTS = [TestPlotFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
