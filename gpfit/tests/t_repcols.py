import unittest
from numpy import array
from gpfit.repcols import repcols

class t_repcols(unittest.TestCase):

    # def test_works_for_a_1D_array_size_n(self):
    #     a = array([1,2,3])
    #     b = array([1,1,2,2,3,3])
    #     self.assertTrue((repcols(a,2) == b).all())

    def test_2D_row_vector(self):
        a = array([[1,2,3]])
        b = array([[1,1,1,2,2,2,3,3,3]])
        self.assertTrue((repcols(a,3) == b).all())

    def test_2D_column_vector(self):
        a = array([[1],[2],[3]])
        b = array([[1,1],[2,2],[3,3]])
        self.assertTrue((repcols(a,2) == b).all())

    def test_2D_matrix(self):
        a = array([[1,2,3],[4,5,6],[7,8,9]])
        b = array([[1,1,2,2,3,3],[4,4,5,5,6,6],[7,7,8,8,9,9]])
        self.assertTrue((repcols(a,2) == b).all())

tests = [t_repcols]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
