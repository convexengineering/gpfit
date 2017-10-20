"unit tests for max_affine_init function"
import unittest
import numpy as np
from numpy import arange, newaxis, vstack, log, exp
from numpy.random import random_sample
from gpfit.max_affine_init import max_affine_init
from .seed import SEED

class TestMaxAffineInitK2(unittest.TestCase):

    x = arange(0., 16.)[:, newaxis]
    y = arange(0., 16.)[:, newaxis]
    K = 2
    ba = max_affine_init(x, y, K)

    def test_ba_ndim_k2(self):
        self.assertEqual(self.ba.ndim, 2)

    def test_ba_shape_k2(self):

        _, dimx = self.x.shape
        self.assertEqual(self.ba.shape, (dimx+1, self.K))

class TestMaxAffineInitK4(unittest.TestCase):
    '''
    This unit test ensures that max affine init produces an array
    of the expected shape and size
    '''
    np.random.seed(SEED)
    Vdd = random_sample(1000,) + 1
    Vth = 0.2*random_sample(1000,) + 0.2
    P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
    u = vstack((Vdd, Vth))
    x = log(u)
    y = log(P)
    x = x.T
    y = y.reshape(y.size, 1)
    K = 4

    ba = max_affine_init(x,y,K)

    def test_ba_shape_k4(self):
        self.assertEqual(self.ba.shape, (3, 4))



tests = [TestMaxAffineInitK2,
         TestMaxAffineInitK4]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
