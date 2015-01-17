import unittest
from gpfit.implicit_softmax_affine import implicit_softmax_affine
from numpy import arange, newaxis

class t_implicit_softmax_affine(unittest.TestCase):

	x = arange(0.,16.)[:,newaxis]
	params = arange(1.,7.)

	y, dydp = implicit_softmax_affine(x,params) 

	def test_y_size(self):
		self.assertEqual(self.y.size, self.x.size)

	def test_y_ndim(self):
		self.assertEqual(self.y.ndim, 1)

	def test_dydp_shape(self):
		self.assertEqual(self.dydp.shape, (self.x.size, self.params.size))

	def test_dydp_ndim(self):
		self.assertEqual(self.dydp.ndim, 2)

tests = [t_implicit_softmax_affine]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)