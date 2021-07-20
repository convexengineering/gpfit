"unit tests for plot fit methods"
import unittest
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestPlot(unittest.TestCase):
    "Unit tests for plot methods"

    np.random.seed(SEED)
    N = 51
    u = np.logspace(0, np.log10(3), N)
    w = (u**2 + 3)/(u + 1)**2
    x = np.log(u)
    y = np.log(w)
    K = 2

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot()
        fig.savefig("plots/ma_test.png")

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot()
        fig.savefig("plots/sma_test.png")

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot()
        fig.savefig("plots/isma_test.png")


class TestPlotSurface(unittest.TestCase):
    "Unit tests for plot_surface methods"

    np.random.seed(SEED)
    Vdd = np.random.random_sample(100) + 1
    Vth = 0.2*np.random.random_sample(100) + 0.2
    P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)
    u = np.vstack((Vdd, Vth))
    x = np.log(u)
    y = np.log(P)
    K = 3

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_surface(azim=135)
        fig.savefig("plots/ma_test_surface.png")

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_surface(azim=135)
        fig.savefig("plots/sma_test_surface.png")

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_surface(azim=135)
        fig.savefig("plots/isma_test_surface.png")


class TestPlotSlices(unittest.TestCase):
    "Unit tests for plot_slices method"

    np.random.seed(SEED)
    Vdd = np.linspace(1, 2, 10)
    Vth = np.linspace(0.2, 0.4, 5)
    Vdd, Vth = np.meshgrid(Vdd, Vth)
    Vdd, Vth = Vdd.flatten(), Vth.flatten()
    P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)
    u = np.vstack((Vdd, Vth))
    x = np.log(u)
    y = np.log(P)
    K = 3

    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_slices()
        fig.savefig("plots/ma_test_slices.png")

    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_slices()
        fig.savefig("plots/sma_test_slices.png")

    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K)
        fig, _ = f.plot_slices()
        fig.savefig("plots/isma_test_slices.png")


TESTS = [
    TestPlot,
    TestPlotSurface,
    TestPlotSlices,
]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
