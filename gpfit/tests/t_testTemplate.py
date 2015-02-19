import unittest

class t_testFamily1(unittest.TestCase):

	def testname1(self):
		self.assertEqual(1, 1)
		self.assertFalse(False)
		self.assertNotEqual(1, 2)
		self.assertTrue(True)

	def testname2(self):
            pass

class t_testFamily2(unittest.TestCase):

	def testname3(self):
		pass

	def testname4(self):
		pass



tests = [t_testFamily1, t_testFamily2]

if __name__ == '__main__':
	suite = unittest.TestSuite()
	loader = unittest.TestLoader()

	for t in tests:
		suite.addTests(loader.loadTestsFromTestCase(t))

	unittest.TextTestRunner(verbosity=2).run(suite)
