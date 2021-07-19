"unit tests for plot fit methods"
import unittest
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestPlotFit(unittest.TestCase):
    "Unit tests for plot_fit methods"

    np.random.seed(SEED)
    N = 51
    u = np.logspace(0, np.log10(3), N)
    w = (u**2 + 3)/(u + 1)**2
    x = np.log(u)
    y = np.log(w)
    K = 2

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_fit()
        fig.savefig("plots/ma_test.png")

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K)
        f.plot_fit()
        fig, _ = f.plot_fit()
        fig.savefig("plots/sma_test.png")

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        f.plot_fit()
        fig, _ = f.plot_fit()
        fig.savefig("plots/isma_test.png")


TESTS = [TestPlotFit]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
