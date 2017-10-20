import unittest
from gpfit.lse_scaled import lse_scaled
from numpy import arange, newaxis

class t_lse_scaled(unittest.TestCase):

    x = arange(1.,31.).reshape(15,2)
    alpha = 1.
    y, dydx, dydalpha = lse_scaled(x,alpha)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.shape[0])

    def test_dydx_ndim(self):
        self.assertEqual(self.dydx.ndim, 2)

    def test_dydx_shape_0(self):
        self.assertEqual(self.dydx.shape[0], self.x.shape[0])

    def test_dydx_shape_1(self):
        pass
        # self.assertEqual(self.dydx.shape[0], ???????)

    def test_dydalpha_ndim(self):
        self.assertEqual(self.dydalpha.ndim, 1)

    def test_dydalpha_size(self):
        self.assertEqual(self.dydalpha.size, self.x.shape[0])

    # test alpha is integer? negative? 0? array?


tests = [t_lse_scaled]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
