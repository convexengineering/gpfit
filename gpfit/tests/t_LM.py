"Tests LM"
import unittest
from numpy import arange, newaxis
from gpfit.LM import LM
from gpfit.max_affine import max_affine


def rfun(params):
    "A specific residual function."
    [yhat, drdp] = max_affine(arange(0., 16.)[:, newaxis], params)
    r = yhat - arange(0., 16.)[:, newaxis].T[0]
    return r, drdp


class t_LM(unittest.TestCase):
    "Tests LM"
    initparams = arange(1., 5.)
    params, RMStraj = LM(rfun, initparams)

    def test_params_size(self):
        self.assertEqual(self.params.size, self.initparams.size)

    def test_params_ndim(self):
        self.assertEqual(self.params.ndim, 1)

    def test_RMStraj_shape(self):
        # self.assertEqual(self.RMStraj.shape, (self.x.size, self.ba.size))
        pass

    def test_RMStraj_ndim(self):
        self.assertEqual(self.RMStraj.ndim, 1)

TESTS = [t_LM]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in TESTS:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
