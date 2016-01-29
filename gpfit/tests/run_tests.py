import unittest
tests = []

import t_LM
tests += t_LM.tests

import t_generic_resid_fun
tests += t_generic_resid_fun.tests

import t_lse_implicit
tests += t_lse_implicit.tests

import t_lse_scaled
tests += t_lse_scaled.tests

import t_max_affine_init
tests += t_max_affine_init.tests

import t_max_affine
tests += t_max_affine.tests

import t_softmax_affine
tests += t_softmax_affine.tests

import t_implicit_softmax_affine
tests += t_implicit_softmax_affine.tests

import t_print_fit
tests += t_print_fit.tests

import t_ex6_1
tests += t_ex6_1.tests

import t_ex6_3
tests += t_ex6_3.tests

from gpfit.tests import t_examples
tests += t_examples.TESTS

def run():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run()
