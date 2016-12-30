"Test max_affine"
import unittest
from numpy import arange, newaxis
from gpfit.max_affine import max_affine


class t_max_affine(unittest.TestCase):
    "Test max_affine"

    x = arange(0., 16.)[:, newaxis]
    ba = arange(1., 7.).reshape(2, 3)

    y, dydba = max_affine(x, ba)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.size)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_dydba_shape(self):
        self.assertEqual(self.dydba.shape, (self.x.size, self.ba.size))

    def test_dydba_ndim(self):
        self.assertEqual(self.dydba.ndim, 2)

TESTS = [t_max_affine]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
