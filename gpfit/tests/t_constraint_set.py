"unit tests for gpfit.print_fit module"
import unittest
import numpy as np
from gpkit import Variable, Model
from gpfit.fit import SoftmaxAffine

SEED = 33404


class TestFitConstraintSet(unittest.TestCase):
    "Unit tests for FitConstraintSet"

    u = np.logspace(0, np.log10(3), 501)
    w = (u**2 + 3)/(u + 1)**2
    x = np.log(u)
    y = np.log(w)
    K = 3
    f = SoftmaxAffine(x, y, K, seed=SEED)
    uvar = Variable("u")
    wvar = Variable("w")
    fcs = f.constraint_set(ivar=wvar, dvars=[uvar])

    def test_fit_constraint_set(self):
        m = Model(self.wvar, [self.fcs, self.uvar <= 2])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 0.7777593)

    def test_outside_bounds(self):
        m = Model(self.wvar, [self.fcs, self.uvar <= 4])
        sol = m.solve(verbosity=0)
        self.assertTrue(len(sol['warnings']["Fit Out-of-Bounds"]) == 1)
        self.assertAlmostEqual(sol["cost"], 0.7499633)


TESTS = [TestFitConstraintSet]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
