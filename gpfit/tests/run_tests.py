import unittest
TESTS = []

import t_LM
TESTS += t_LM.tests

import t_generic_resid_fun
TESTS += t_generic_resid_fun.tests

import t_lse_implicit
TESTS += t_lse_implicit.tests

import t_lse_scaled
TESTS += t_lse_scaled.tests

import t_max_affine_init
TESTS += t_max_affine_init.tests

import t_max_affine
TESTS += t_max_affine.tests

import t_softmax_affine
TESTS += t_softmax_affine.tests

import t_implicit_softmax_affine
TESTS += t_implicit_softmax_affine.tests

import t_print_fit
TESTS += t_print_fit.tests

import t_ex6_1
TESTS += t_ex6_1.tests

import t_ex6_3
TESTS += t_ex6_3.tests

import t_examples
TESTS += t_examples.TESTS

def run():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in TESTS:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run()
