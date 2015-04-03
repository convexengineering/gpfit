import unittest
from numpy import arange, newaxis
from gpfit.max_affine import max_affine
from gpfit.generic_resid_fun import generic_resid_fun

class t_generic_resid_fun(unittest.TestCase):

    yfun = max_affine
    xdata = arange(1.,11.)[:,newaxis]
    ydata = arange(1.,11.)[:,newaxis]
    params = arange(1.,5.)
    r, drdp = generic_resid_fun(yfun, xdata, ydata, params)

    def test_size_of_r(self):
        self.assertTrue(self.r.size == self.xdata.size)

    def test_dimension_of_r(self):
        self.assertTrue(self.r.ndim == 1)

    def test_size_of_drdp(self):
        self.assertTrue(self.drdp.shape == (self.xdata.size, self.params.size))

    def test_dimension_of_drdp(self):
        self.assertTrue(self.drdp.ndim == 2)


tests = [t_generic_resid_fun]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
