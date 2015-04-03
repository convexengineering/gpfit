import unittest
from gpfit.max_affine_init import max_affine_init
from numpy import arange, newaxis

class t_max_affine_init_K_equals_2(unittest.TestCase):

    x = arange(0.,16.)[:,newaxis]
    y = arange(0.,16.)[:,newaxis]
    K = 2
    ba = max_affine_init(x, y, K)

    def test_ba_ndim(self):
        self.assertEqual(self.ba.ndim, 2)

    def test_ba_shape(self):
        
        npt,dimx = self.x.shape

        self.assertEqual(self.ba.shape, (dimx+1 ,self.K))

class t_max_affine_init_other_K(unittest.TestCase):

    def test_K_as_an_array_of_ks(self):
        pass

    def test_K_greater_than_2(self):
        pass



tests = [t_max_affine_init_K_equals_2]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
