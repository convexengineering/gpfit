"Test evaluate methods"
import unittest
from numpy import arange, newaxis
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine


class TestMaxAffine(unittest.TestCase):
    "Test max_affine"

    x = arange(0.0, 16.0)[:, newaxis]
    ba = arange(1.0, 7.0).reshape(2, 3)

    y, dydba = MaxAffine.evaluate(x, ba)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.size)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_dydba_shape(self):
        self.assertEqual(self.dydba.shape, (self.x.size, self.ba.size))

    def test_dydba_ndim(self):
        self.assertEqual(self.dydba.ndim, 2)


class TestSoftmaxAffine(unittest.TestCase):
    "Tests softmax_affine"

    x = arange(0.0, 16.0)[:, newaxis]
    params = arange(1.0, 6.0)

    y, dydp = SoftmaxAffine.evaluate(x, params)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.size)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_dydp_shape(self):
        self.assertEqual(self.dydp.shape, (self.x.size, self.params.size))

    def test_dydp_ndim(self):
        self.assertEqual(self.dydp.ndim, 2)


class TestImplicitSoftmaxAffine(unittest.TestCase):
    "Tests implicit_softmax_affine"

    x = arange(0.0, 16.0)[:, newaxis]
    params = arange(1.0, 7.0)

    y, dydp = ImplicitSoftmaxAffine.evaluate(x, params)

    def test_y_size(self):
        self.assertEqual(self.y.size, self.x.size)

    def test_y_ndim(self):
        self.assertEqual(self.y.ndim, 1)

    def test_dydp_shape(self):
        self.assertEqual(self.dydp.shape, (self.x.size, self.params.size))

    def test_dydp_ndim(self):
        self.assertEqual(self.dydp.ndim, 2)


TESTS = [
    TestMaxAffine,
    TestSoftmaxAffine,
    TestImplicitSoftmaxAffine,
]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
