"unit tests based on example 6.3 from Hoburg/Abbeel GPfit paper"
import unittest
import numpy as np
from numpy import log, exp, vstack
from numpy.random import random_sample
from .seed import SEED
from gpfit.fit import fit

class TestEx63ISMA(unittest.TestCase):
    '''
    ISMA unit tests based on example 6.3 from GPfit paper
    '''
    np.random.seed(SEED)
    Vdd = random_sample(1000,) + 1
    Vth = 0.2*random_sample(1000,) + 0.2
    P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
    u = vstack((Vdd,Vth))
    x = log(u)
    y = log(P)
    K = 4

    cstrt, rms_error = fit(x, y, K, "ISMA")

    def test_rms_error(self):
        self.assertTrue(self.rms_error < 5e-4)

class TestEx63SMA(unittest.TestCase):
    '''
    SMA unit tests based on example 6.3 from GPfit paper
    '''
    np.random.seed(SEED)
    Vdd = random_sample(1000,) + 1
    Vth = 0.2*random_sample(1000,) + 0.2
    P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
    u = vstack((Vdd,Vth))
    x = log(u)
    y = log(P)
    K = 4

    cstrt, rms_error = fit(x, y, K, "SMA")

    def test_rms_error(self):
        self.assertTrue(self.rms_error < 5e-4)

class TestEx63MA(unittest.TestCase):
    '''
    MA unit tests based on example 6.3 from GPfit paper
    '''
    np.random.seed(SEED)
    Vdd = random_sample(1000,) + 1
    Vth = 0.2*random_sample(1000,) + 0.2
    P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
    u = vstack((Vdd,Vth))
    x = log(u)
    y = log(P)
    K = 4

    cstrt, rms_error = fit(x, y, K, "MA")

    def test_rms_error(self):
        self.assertTrue(self.rms_error < 1e-2)


tests = [TestEx63ISMA,
         TestEx63SMA,
         TestEx63MA]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
