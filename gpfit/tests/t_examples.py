import unittest
import numpy as np
import sys
import os
import importlib


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

class StdoutCaptured(object):
    def __init__(self, logfilename=None):
        self.logfilename = logfilename

    def __enter__(self):
        self.original_stdout = sys.stdout
        if self.logfilename:
            filepath = EXAMPLE_DIR+os.sep+"%s_output.txt" % self.logfilename
            logfile = open(filepath, "w")
        else:
            logfile = NullFile()
        sys.stdout = logfile

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self.original_stdout
    
def new_test(name):
    def test(self):
        logfilename = name if name not in IMPORTED_EXAMPLES else None
        with StdoutCaptured(logfilename):
            if name not in IMPORTED_EXAMPLES:
                IMPORTED_EXAMPLES[name] = importlib.import_module(name)
            else:
                reload(IMPORTED_EXAMPLES[name])
        getattr(self, name)(IMPORTED_EXAMPLES[name])
    return test

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
            setattr(TestExamples, new_name, new_test(name))
    TESTS.append(TestExamples)

if __name__ == "__main__":
    from gpfit.tests.helpers import run_tests
    run_tests(TESTS)
