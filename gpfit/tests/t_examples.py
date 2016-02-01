import unittest
import numpy as np
import sys
import os
import importlib
from gpkit.tests.t_examples import StdoutCaptured, new_test 

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR+'../../../docs/source/examples')
IMPORTED_EXAMPLES = {}


class TestExamples(unittest.TestCase):

    # To test a new example, add a function called `test_$EXAMPLENAME`, where
    # $EXAMPLENAME is the name of your example in docs/source/examples without
    # the file extension.
    #
    # This function should accept two arguments (e.g. 'self' and 'example').
    # The imported example script will be passed to the second: anything that
    # was a global variable (e.g, "sol") in the original script is available
    # as an attribute (e.g., "example.sol")
    #
    # If you don't want to perform any checks on the example besides making
    # sure it runs, just put "pass" as the function's body, e.g.:
    #
    #       def test_dummy_example(self, example):
    #           pass
    #
    # But it's good practice to ensure the example's solution as well, e.g.:
    #
    #       def test_dummy_example(self, example):
    #           self.assertAlmostEqual(example.sol["cost"], 3.121)
    #

    def test_t_ex6_1(self, example):
        pass

TESTS = []
if os.path.isdir(EXAMPLE_DIR):
    sys.path.insert(0, EXAMPLE_DIR)
    for fn in dir(TestExamples):
        if fn[:5] == "test_":
            name = fn[5:]
            old_test = getattr(TestExamples, fn)
            setattr(TestExamples, name, old_test)  # move to a non-test fn
            delattr(TestExamples, fn)  # delete the old old_test
            new_name = "test_%s" % (name)
            setattr(TestExamples, new_name, new_test(name,None))
    TESTS.append(TestExamples)

if __name__ == "__main__":
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
