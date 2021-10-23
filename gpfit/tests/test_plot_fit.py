"unit tests for plot fit methods"
import pytest
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404


class TestPlot:
    "Unit tests for plot methods"

    N = 51
    u = np.logspace(0, np.log10(3), N)
    w = (u**2 + 3)/(u + 1)**2
    x = np.log(u)
    y = np.log(w)
    K = 2

    @pytest.mark.mpl_image_compare(filename='ma_1d.png')
    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot()
        return fig

    @pytest.mark.mpl_image_compare(filename='sma_1d.png')
    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot()
        return fig

    @pytest.mark.mpl_image_compare(filename='isma_1d.png')
    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot()
        return fig


class TestPlotSurface:
    "Unit tests for plot_surface methods"

    rng = np.random.RandomState(33404)
    Vdd = rng.random_sample(100) + 1
    Vth = 0.2*rng.random_sample(100) + 0.2
    P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)
    u = np.vstack((Vdd, Vth))
    x = np.log(u)
    y = np.log(P)
    K = 3

    @pytest.mark.mpl_image_compare(filename='ma_2d_surface.png')
    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_surface(azim=135)
        return fig

    @pytest.mark.mpl_image_compare(filename='sma_2d_surface.png')
    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_surface(azim=135)
        return fig

    @pytest.mark.mpl_image_compare(filename='isma_2d_surface.png')
    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_surface(azim=135)
        return fig


class TestPlotSlices:
    "Unit tests for plot_slices method"

    Vdd = np.linspace(1, 2, 10)
    Vth = np.linspace(0.2, 0.4, 5)
    Vdd, Vth = np.meshgrid(Vdd, Vth)
    Vdd, Vth = Vdd.flatten(), Vth.flatten()
    P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)
    u = np.vstack((Vdd, Vth))
    x = np.log(u)
    y = np.log(P)
    K = 3

    @pytest.mark.mpl_image_compare(filename='ma_2d_slices.png')
    def test_max_affine(self):
        f = MaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_slices()
        return fig

    @pytest.mark.mpl_image_compare(filename='sma_2d_slices.png')
    def test_softmax_affine(self):
        f = SoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_slices()
        return fig

    @pytest.mark.mpl_image_compare(filename='isma_2d_slices.png')
    def test_implicit_softmax_affine(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, self.K, seed=SEED)
        fig, _ = f.plot_slices()
        return fig

TESTS = [
    TestPlot,
    TestPlotSurface,
    TestPlotSlices,
]
