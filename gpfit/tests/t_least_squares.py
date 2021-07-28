"Tests levenberg_marquardt"
import unittest
from numpy import arange, newaxis
from gpfit.maths.least_squares import levenberg_marquardt
from gpfit.fit import MaxAffine


def rfun(params):
    "A specific residual function."
    [yhat, drdp] = MaxAffine.evaluate(arange(0.0, 16.0)[:, newaxis], params)
    r = yhat - arange(0.0, 16.0)[:, newaxis].T[0]
    return r, drdp


class t_levenberg_marquardt(unittest.TestCase):
    "Tests levenberg_marquardt"
    initparams = arange(1.0, 5.0)
    params, RMStraj = levenberg_marquardt(rfun, initparams)

    def test_params_size(self):
        self.assertEqual(self.params.size, self.initparams.size)

    def test_params_ndim(self):
        self.assertEqual(self.params.ndim, 1)

    def test_rmstraj_shape(self):
        # self.assertEqual(self.RMStraj.shape, (self.x.size, self.ba.size))
        pass

    def test_rmstraj_ndim(self):

        self.assertEqual(self.RMStraj.ndim, 1)


TESTS = [t_levenberg_marquardt]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
