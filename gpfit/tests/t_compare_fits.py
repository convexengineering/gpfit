import unittest
from numpy import arange, array, vstack, newaxis, allclose
from gpfit.compare_fits import compare_fits

class t_max_affine(unittest.TestCase):

	def test_max_affine_1(self):
		x = arange(0.,16.)[:,newaxis]
		y1 = arange(0.,11.)[:,newaxis]
		y2 = arange(12.,21.,2.)[:,newaxis]
		y = vstack((y1,y2))
		Ks = array([2])
		ntry = array([1])
		s = compare_fits(x,y,Ks,ntry)
		max_affine_params = s['max_affine']['params'][0][0]

		self.assertTrue(
						allclose(max_affine_params, array([0,1,-10,2]))
						or
						allclose(max_affine_params, array([-10,2,0,1]))
						)

class t_softmax_optMAinit(unittest.TestCase):

	def test_softmax_optMAinit_1(self):
		x = arange(0.,16.)[:,newaxis]
		y1 = arange(0.,11.)[:,newaxis]
		y2 = arange(12.,21.,2.)[:,newaxis]
		y = vstack((y1,y2))
		Ks = array([2])
		ntry = array([1])
		s = compare_fits(x,y,Ks,ntry)
		softmax_optMA_params = s['softmax_optMAinit']['params'][0][0]

		self.assertTrue(
						allclose(softmax_optMA_params, array([-9.4918,1.9652,-0.0076,1.0035,0.0035]), atol = 1E-4)
						or
						allclose(softmax_optMA_params, array([-0.0076,1.0035,-9.4918,1.9652,0.0035]), atol = 1E-4)
						)

# class t_softmax_originit(unittest.TestCase):

# 	def test_softmax_originit_1(self):
# 		x = arange(0.,16.)[:,newaxis]
# 		y1 = arange(0.,11.)[:,newaxis]
# 		y2 = arange(12.,21.,2.)[:,newaxis]
# 		y = vstack((y1,y2))
# 		Ks = array([2])
# 		ntry = array([1])
# 		s = compare_fits(x,y,Ks,ntry)
# 		softmax_orig_params = s['softmax_originit']['params'][0][0]

# 		self.assertTrue(
# 						allclose(softmax_orig_params, array([-9.4918,1.9652,-0.0076,1.0035,0.0035]))
# 						or
# 						allclose(softmax_orig_params, array([-0.0076,1.0035,-9.4918,1.9652,0.0035]))
# 						)

# class t_implicit_originit(unittest.TestCase):

# 	def test_softmax_originit_1(self):
# 		x = arange(0.,16.)[:,newaxis]
# 		y1 = arange(0.,11.)[:,newaxis]
# 		y2 = arange(12.,21.,2.)[:,newaxis]
# 		y = vstack((y1,y2))
# 		Ks = array([2])
# 		ntry = array([1])
# 		s = compare_fits(x,y,Ks,ntry)
# 		implicit_orig_params = s['implicit_originit']['params'][0][0]

# 		self.assertTrue(
# 						allclose(implicit_orig_params, array([0,1,-10,2]))
# 						or
# 						allclose(implicit_orig_params, array([-10,2,0,1]))
# 						)

tests = [t_max_affine,
		 t_softmax_optMAinit]
		 # t_softmax_originit,
		 # t_implicit_originit]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)