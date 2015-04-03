import unittest
from numpy import arange, array, vstack, newaxis, allclose
from gpfit.compare_fits import compare_fits

class t_max_affine(unittest.TestCase):

    x = arange(0.,16.)[:,newaxis]
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

    def test_max_affine_1(self):

        max_affine_params = self.s['max_affine']['params'][0][0]

        self.assertTrue(
                        allclose(max_affine_params, array([0,1,-10,2]))
                        or
                        allclose(max_affine_params, array([-10,2,0,1]))
                        )

class t_softmax_optMAinit(unittest.TestCase):

    x = arange(0.,16.)[:,newaxis]
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

    def test_softmax_optMAinit_1(self):

        softmax_optMA_params = self.s['softmax_optMAinit']['params'][0][0]

        self.assertTrue(
                        allclose(softmax_optMA_params,
                                 array([-9.4918,1.9652,-0.0076,1.0035,0.0035]),
                                 atol = 1E-4)
                        or
                        allclose(softmax_optMA_params,
                                 array([-0.0076,1.0035,-9.4918,1.9652,0.0035]),
                                 atol = 1E-4)
                        )

# class t_softmax_originit(unittest.TestCase):

    # x = arange(0.,16.)[:,newaxis]
    # y1 = arange(0.,11.)[:,newaxis]
    # y2 = arange(12.,21.,2.)[:,newaxis]
    # y = vstack((y1,y2))
    # Ks = array([2])
    # ntry = array([1])
    # s = compare_fits(x,y,Ks,ntry)

#     def test_softmax_originit_1(self):

#         softmax_orig_params = self.s['softmax_originit']['params'][0][0]

#         self.assertTrue(
#                         allclose(softmax_orig_params, array([-9.4918,1.9652,-0.0076,1.0035,0.0035]))
#                         or
#                         allclose(softmax_orig_params, array([-0.0076,1.0035,-9.4918,1.9652,0.0035]))
#                         )

# class t_implicit_originit(unittest.TestCase):

    # x = arange(0.,16.)[:,newaxis]
    # y1 = arange(0.,11.)[:,newaxis]
    # y2 = arange(12.,21.,2.)[:,newaxis]
    # y = vstack((y1,y2))
    # Ks = array([2])
    # ntry = array([1])
    # s = compare_fits(x,y,Ks,ntry)

#     def test_softmax_originit_1(self):

#         implicit_orig_params = self.s['implicit_originit']['params'][0][0]

#         self.assertTrue(
#                         allclose(implicit_orig_params, array([0,1,-10,2]))
#                         or
#                         allclose(implicit_orig_params, array([-10,2,0,1]))
#                         )

class t_k_tests(unittest.TestCase):

    x = arange(0.,16.)[:,newaxis]
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    ntry = 1

    def test_k_3(self):
        Ks = array([3])
        s = compare_fits(self.x, self.y, Ks, self.ntry)

    def test_k_multi_element_array(self):
        Ks = array([1,3])
        s = compare_fits(self.x, self.y, Ks, self.ntry)

    def test_k_is_not_array(self):
        k = 2
        s = compare_fits(self.x, self.y, k, self.ntry)


tests = [t_max_affine,
         t_softmax_optMAinit,
         # t_softmax_originit,
         # t_implicit_originit,
         t_k_tests]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
