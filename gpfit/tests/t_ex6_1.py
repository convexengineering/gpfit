import unittest
from numpy import logspace, log, exp, log10
from gpfit.fit import fit

class t_ex6_1_ISMA(unittest.TestCase):
    def test_rms_error(self):
        self.asserdtTrue(self.rms_error < 1e-5)
        '''
        ISMA unit tests based on example 6.1 from GPfit paper
        '''
        m = 501
        u = logspace(0,log10(3),501)
        w = (u**2 + 3)/(u+1)**2
        x = log(u)
        y = log(w)
        K = 3

        cstrt, rms_error = fit(x, y, K, "ISMA")


class t_ex6_1_SMA(unittest.TestCase):
    '''
    SMA unit tests based on example 6.1 from GPfit paper
    '''
    m = 501
    u = logspace(0,log10(3),501)
    w = (u**2 + 3)/(u+1)**2
    x = log(u)
    y = log(w)
    K = 3

    cstrt, rms_error = fit(x, y, K, "SMA")

    def test_rms_error(self):
        self.assertTrue(self.rms_error < 1e-4)

class t_ex6_1_MA(unittest.TestCase):
    '''
    MA unit tests based on example 6.1 from GPfit paper
    '''
    m = 501
    u = logspace(0,log10(3),501)
    w = (u**2 + 3)/(u+1)**2
    x = log(u)
    y = log(w)
    K = 3

    cstrt, rms_error = fit(x, y, K, "MA")

    def test_rms_error(self):
        self.assertTrue(self.rms_error < 1e-2)


tests = [t_ex6_1_ISMA,
         t_ex6_1_SMA,
         t_ex6_1_MA]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
