"Runs all tests"
from gpkit.tests.helpers import run_tests

TESTS = []

from gpfit.tests import t_least_squares
TESTS += t_least_squares.TESTS

from gpfit.tests import t_logsumexp
TESTS += t_logsumexp.TESTS

from gpfit.tests import t_initialize
TESTS += t_initialize.TESTS

from gpfit.tests import t_evaluate
TESTS += t_evaluate.TESTS

from gpfit.tests import t_print_fit
TESTS += t_print_fit.TESTS

from gpfit.tests import t_plot_fit
TESTS += t_plot_fit.TESTS

#from gpfit.tests import t_examples
#TESTS += t_examples.TESTS

from gpfit.tests import t_fit
TESTS += t_fit.TESTS

from gpfit.tests import t_constraint_set
TESTS += t_constraint_set.TESTS


def run(xmloutput=False):
    """Run all gpfit unit tests.

    Arguments
    ---------
    xmloutput: bool
        If true, generate xml output files for continuous integration
    """
    if xmloutput:
        run_tests(TESTS, xmloutput="test_reports")
    else:
        run_tests(TESTS)


if __name__ == "__main__":
    run()
