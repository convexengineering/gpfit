"""unit tests for printing functionality"""
import unittest
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

A = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12])
B = np.array([1, 5, 9])
ALPHA = np.array([1/13, 1/14, 1/15])
DIM = 3
K = 3


class TestPrintFit(unittest.TestCase):
    "Unit tests for print functions"
    u = np.logspace(0, np.log10(3), 501)
    w = (u**2 + 3)/(u + 1)**2
    x = np.log(u)
    y = np.log(w)

    def test_ma(self):
        f = MaxAffine(self.x, self.y, K)
        f.A = A
        f.B = B
        f.d = DIM
        strings = f.__repr__()
        self.assertEqual(
            strings,
            (
                "w = 2.71828*(u_1)^2*(u_2)^3*(u_3)^4\n"
                "w = 148.413*(u_1)^6*(u_2)^7*(u_3)^8\n"
                "w = 8103.08*(u_1)^10*(u_2)^11*(u_3)^12"
            ),
        )

    def test_sma(self):
        f = SoftmaxAffine(self.x, self.y, K)
        f.A = A
        f.B = B
        f.alpha = ALPHA[0]
        f.d = DIM
        strings = f.__repr__()
        self.assertEqual(
            strings,
            (
                "w^0.0769231 = 1.07996*(u_1)^0.153846*"
                "(u_2)^0.230769*(u_3)^0.307692\n"
                "    + 1.46905*(u_1)^0.461538*"
                "(u_2)^0.538462*(u_3)^0.615385\n"
                "    + 1.99832*(u_1)^0.769231*"
                "(u_2)^0.846154*(u_3)^0.923077"
            ),
        )

    def test_isma(self):
        f = ImplicitSoftmaxAffine(self.x, self.y, K)
        f.A = A
        f.B = B
        f.alpha = ALPHA
        f.d = DIM
        strings = f.__repr__()
        self.assertEqual(
            strings,
            (
                "1 = (1.07996/w^0.0769231)*(u_1)^0.153846*"
                "(u_2)^0.230769*(u_3)^0.307692\n"
                "  + (1.42924/w^0.0714286)*(u_1)^0.428571*"
                "(u_2)^0.5*(u_3)^0.571429\n"
                "  + (1.82212/w^0.0666667)*(u_1)^0.666667*"
                "(u_2)^0.733333*(u_3)^0.8"
            ),
        )


TESTS = [TestPrintFit]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
