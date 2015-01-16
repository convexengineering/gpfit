import unittest
tests = []

import t_repcols
tests += t_repcols.tests

import t_generic_resid_fun
tests += t_generic_resid_fun.tests

import t_compare_fits
tests +=t_compare_fits.tests

def run():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run()