import unittest
from gpfit.LM import LM
from gpfit.max_affine import max_affine
from gpfit.generic_resid_fun import generic_resid_fun
from numpy import arange, newaxis

class t_LM(unittest.TestCase):
    
    def rfun (p): return generic_resid_fun(max_affine,
                                            arange(0.,16.)[:,newaxis],
                                            arange(0.,16.)[:,newaxis],
                                            p)
    residfun = rfun
    initparams = arange(1.,5.)

    params, RMStraj = LM(residfun, initparams) 

    def test_params_size(self):
        self.assertEqual(self.params.size, self.initparams.size)

    def test_params_ndim(self):
        self.assertEqual(self.params.ndim, 1)

    def test_RMStraj_shape(self):
        # self.assertEqual(self.RMStraj.shape, (self.x.size, self.ba.size))
        pass

    def test_RMStraj_ndim(self):
        self.assertEqual(self.RMStraj.ndim, 2)

tests = [t_LM]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
