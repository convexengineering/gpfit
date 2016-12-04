import unittest
from gpfit.print_fit import print_MA, print_SMA, print_ISMA
from numpy import array, arange

class t_print_MA(unittest.TestCase):

    A = array([ 2,  3,  4,  6,  7,  8, 10, 11, 12])
    B = array([1, 5, 9])
    d = 3
    K = 3
    stringList = print_MA(A, B, d, K)

    def test_MA(self):
        self.assertEqual(self.stringList,
            ['w = 2.71828 * (u_1)**2 * (u_2)**3 * (u_3)**4',
             'w = 148.413 * (u_1)**6 * (u_2)**7 * (u_3)**8',
             'w = 8103.08 * (u_1)**10 * (u_2)**11 * (u_3)**12'])

class t_print_SMA(unittest.TestCase):

    A = array([ 2,  3,  4,  6,  7,  8, 10, 11, 12])
    B = array([1, 5, 9])
    alpha = 1/13.0
    d = 3
    K = 3
    stringList = print_SMA(A, B, alpha, d, K)

    def test_SMA(self):
        self.assertEqual(self.stringList,
            ['w**0.0769231 = 1.07996 * (u_1)**0.153846 * (u_2)**0.230769 * (u_3)**0.307692',
             '    + 1.46905 * (u_1)**0.461538 * (u_2)**0.538462 * (u_3)**0.615385',
             '    + 1.99832 * (u_1)**0.769231 * (u_2)**0.846154 * (u_3)**0.923077'])

class t_print_ISMA(unittest.TestCase):
    
    A = array([ 2,  3,  4,  6,  7,  8, 10, 11, 12])
    B = array([1, 5, 9])
    alpha = array([ 1./13.,  1./14.,  1./15.])
    d = 3
    K = 3
    stringList = print_ISMA(A, B, alpha, d, K)

    def test_ISMA(self):
        self.assertEqual(self.stringList,
            ['1 = (1.07996/w**0.0769231) * (u_1)**0.153846 * (u_2)**0.230769 * (u_3)**0.307692',
             '    + (1.42924/w**0.0714286) * (u_1)**0.428571 * (u_2)**0.5 * (u_3)**0.571429',
             '    + (1.82212/w**0.0666667) * (u_1)**0.666667 * (u_2)**0.733333 * (u_3)**0.8'])

tests = [t_print_MA, t_print_SMA, t_print_ISMA]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
