import unittest

class t_testFamily1(unittest.TestCase):

	def test_something1(self):
		self.assertEqual()
		self.assertFalse()
		self.assertNotEqual()
		self.assertTrue()

	def test_something2(self):
		self.assertEqual()
		self.assertFalse()
		self.assertNotEqual()
		self.assertTrue()

class t_testFamily2(unittest.TestCase):

	def test_something3(self):
		pass

	def test_something4(self):
		pass



tests = [t_testFamily1, t_testFamily2]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)