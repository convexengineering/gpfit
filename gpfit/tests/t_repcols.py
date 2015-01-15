import unittest
from numpy import array
from gpfit import repcols

class t_repcols(unittest.TestCase):

	def works_for_a_1D_array_size_n(self):
		a = array([1,2,3])
		b = array([1,1,2,2,3,3])
		self.assertEqual(self(a,2),b)

	def works_for_a_2D_row_vector(self):
		a = array([1,2,3])
		b = array([1,1,1,2,2,2,3,3,3])
		self.assertEqual(self(a,3),b)

	def works_for_a_2D_column_vector(self):
		a = array([[1],[2],[3]])
		b = array([[1,1],[2,2],[3,3]])
		self.assertEqual(self(a,2),b)

	def works_for_a_2D_matrix(self):
		a = array([[1,2,3],[4,5,6],[7,8,9]])
		b = array([[1,1,2,2,3,3],[4,4,5,5,6,6],[7,7,8,8,9,9]])
		self.assertEqual(self(a,2),b)

tests = [t_arrays]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)