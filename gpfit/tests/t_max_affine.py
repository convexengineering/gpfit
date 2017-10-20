import unittest
from gpfit.max_affine import max_affine
from numpy import arange, newaxis

class t_max_affine(unittest.TestCase):

    x = arange(0.,16.)[:,newaxis]
    ba = arange(1.,7.).reshape(2,3)

    y, dydba = max_affine(x,ba) 

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.size)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_dydba_shape(self):
        self.assertEqual(self.dydba.shape, (self.x.size, self.ba.size))

    def test_dydba_ndim(self):
        self.assertEqual(self.dydba.ndim, 2)

tests = [t_max_affine]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
