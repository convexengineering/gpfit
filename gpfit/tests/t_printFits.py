import unittest
from gpfit.printFits import print_max_affine, print_softmax_affine, print_implicit_softmax_affine
from numpy import array, arange

class t_print_max_affine(unittest.TestCase):

	params = arange(1.,5.).reshape(1,4)
	a, b = print_max_affine(params)

	def test_MA_a(self):
		self.assertTrue((self.a == array([[2.,4.]])).all()) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<need to figure out dimension issues

	def test_MA_b(self):
		self.assertTrue((self.b == array([[1.,3.]])).all())

class t_print_softmax_affine(unittest.TestCase):

	params = arange(1.,6.).reshape(1,5)
	a, b, alpha = print_softmax_affine(params)

	def test_SMA_a(self):
		self.assertTrue(all(self.a == array([2.,4.])))

	def test_SMA_b(self):
		self.assertTrue(all(self.b == array([1.,3.])))

	def test_SMA_alpha(self):
		self.assertEqual(self.alpha, 5.)

class t_print_implicit_softmax_affine(unittest.TestCase):
	
	params = arange(1.,7.).reshape(1,6)
	a, b, alpha = print_implicit_softmax_affine(params)

	def test_ISMA_a(self):
		self.assertTrue(all(self.a == array([2.,4.])))

	def test_ISMA_b(self):
		self.assertTrue(all(self.b == array([1.,3.])))

	def test_ISMA_alpha(self):
		self.assertTrue(all(self.alpha == array([5., 6.])))



tests = [t_print_max_affine, t_print_softmax_affine, t_print_implicit_softmax_affine]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)