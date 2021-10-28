"Unit testing of tests in docs/source/examples"
import os
import unittest
import pytest
from gpkit.tests.helpers import generate_example_tests
from gpkit import settings
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR + "../../../docs/source/examples")
EXS_IN_PATH = os.path.isdir(EXAMPLE_DIR)


@pytest.mark.skipif(not EXS_IN_PATH, reason="example directory not in path")
class TestExamples(unittest.TestCase):
    """
    To test a new example, add a function called `test_$EXAMPLENAME`, where
    $EXAMPLENAME is the name of your example in docs/source/examples without
    the file extension.

    This function should accept two arguments (e.g. 'self' and 'example').
    The imported example script will be passed to the second: anything that
    was a global variable (e.g, "sol") in the original script is available
    as an attribute (e.g., "example.sol")

    If you don't want to perform any checks on the example besides making
    sure it runs, just put "pass" as the function's body, e.g.:

          def test_dummy_example(self, example):
              pass

    But it's good practice to ensure the example's solution as well, e.g.:

          def test_dummy_example(self, example):
              self.assertAlmostEqual(example.sol["cost"], 3.121)
    """

    def test_ex1(self, example):
        """test_ex1"""
        self.assertLess(example.fma.errors["rms_rel"], 1e-2)
        self.assertLess(example.fma.errors["rms_log"], 1e-2)
        self.assertLess(example.fsma.errors["rms_rel"], 1e-4)
        self.assertLess(example.fsma.errors["rms_log"], 1e-4)
        self.assertLess(example.fisma.errors["rms_rel"], 1e-5)
        self.assertLess(example.fisma.errors["rms_log"], 1e-5)

    def test_ex2(self, example):
        """test_ex2"""
        self.assertLess(example.fma.errors["rms_rel"], 1e-2)
        self.assertLess(example.fma.errors["rms_log"], 1e-2)
        self.assertLess(example.fsma.errors["rms_rel"], 1e-3)
        self.assertLess(example.fsma.errors["rms_log"], 1e-3)
        self.assertLess(example.fisma.errors["rms_rel"], 1e-3)
        self.assertLess(example.fisma.errors["rms_log"], 1e-3)


# use gpkit.tests.helpers.generate_example_tests default: only default solver
TESTS = generate_example_tests(
    EXAMPLE_DIR, [TestExamples], settings["installed_solvers"]
)


if __name__ == "__main__":
    from gpkit.tests.helpers import run_tests

    run_tests(TESTS)
