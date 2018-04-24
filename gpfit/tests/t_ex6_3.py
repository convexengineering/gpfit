"unit tests based on example 6.3 from Hoburg/Abbeel GPfit paper"
import unittest
import numpy as np
from numpy import log, exp, vstack
from numpy.random import random_sample
from gpfit.fit import fit

SEED = 33404


class TestEx63ISMA(unittest.TestCase):
    "ISMA unit tests based on example 6.3 from GPfit paper"

    def test_rms_error(self):
        np.random.seed(SEED)
        Vdd = random_sample(1000,) + 1
        Vth = 0.2*random_sample(1000,) + 0.2
        P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
        u = vstack((Vdd, Vth))
        x = log(u)
        y = log(P)
        K = 4

        _, rms_error = fit(x, y, K, "ISMA")

        self.assertTrue(rms_error < 1e-3)


class TestEx63SMA(unittest.TestCase):
    "SMA unit tests based on example 6.3 from GPfit paper"

    def test_rms_error(self):
        np.random.seed(SEED)
        Vdd = random_sample(1000,) + 1
        Vth = 0.2*random_sample(1000,) + 0.2
        P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
        u = vstack((Vdd, Vth))
        x = log(u)
        y = log(P)
        K = 4

        _, rms_error = fit(x, y, K, "SMA")

        self.assertTrue(rms_error < 1e-2)


class TestEx63MA(unittest.TestCase):
    "MA unit tests based on example 6.3 from GPfit paper"

    def test_rms_error(self):
        np.random.seed(SEED)
        Vdd = random_sample(1000,) + 1
        Vth = 0.2*random_sample(1000,) + 0.2
        P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
        u = vstack((Vdd, Vth))
        x = log(u)
        y = log(P)
        K = 4

        _, rms_error = fit(x, y, K, "MA")

        self.assertTrue(rms_error < 1e-2)


TESTS = [TestEx63ISMA,
         TestEx63SMA,
         TestEx63MA]

if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
