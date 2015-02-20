import unittest
from gpfit.printFits import print_MA, print_SMA, print_ISMA
from numpy import array, arange

class t_print_MA(unittest.TestCase):

	PAR_MA = arange(1,13)
	K = 3
	stringList = print_MA(PAR_MA, K)

	def test_MA(self):
		self.assertEqual(self.stringList,
			['w = 2.72 * (u_0)**2 * (u_1)**3 * (u_2)**4',
 			'w = 148 * (u_0)**6 * (u_1)**7 * (u_2)**8',
 			'w = 8.1e+03 * (u_0)**10 * (u_1)**11 * (u_2)**12'])

class t_print_SMA(unittest.TestCase):

	PAR_SMA = arange(1,14)
	K = 3
	stringList = print_SMA(PAR_SMA, K)

	def test_SMA(self):
		self.assertEqual(self.stringList,
			['w**0.0769 = 1.08 * (u_0)**0.154 * (u_1)**0.231 * (u_2)**0.308',
 			 '    + 1.47 * (u_0)**0.462 * (u_1)**0.538 * (u_2)**0.615',
			 '    + 2 * (u_0)**0.769 * (u_1)**0.846 * (u_2)**0.923'])

class t_print_ISMA(unittest.TestCase):
	
	PAR_ISMA = arange(1,16)
	K = 3
	stringList = print_ISMA(PAR_ISMA, K)

	def test_ISMA(self):
		self.assertEqual(self.stringList,
			['1 = (1.08/w**0.0769) * (u_0)**0.154 * (u_1)**0.231 * (u_2)**0.308',
 			 '    + (1.43/w**0.0714) * (u_0)**0.429 * (u_1)**0.5 * (u_2)**0.571',
 			 '    + (1.82/w**0.0667) * (u_0)**0.667 * (u_1)**0.733 * (u_2)**0.8'])

tests = [t_print_MA, t_print_SMA, t_print_ISMA]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)