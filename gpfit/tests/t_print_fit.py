"unit tests for gpfit.print_fit module"
import unittest
from numpy import array
from gpfit.print_fit import print_ma, print_sma, print_isma

A = array([2, 3, 4, 6, 7, 8, 10, 11, 12])
B = array([1, 5, 9])
ALPHA = array([1/13, 1/14, 1/15])
DIM = 3
K = 3


class TestPrintFit(unittest.TestCase):
    "Unit tests for print_ma, print_sma, print_isma"

    def test_ma(self):
        strings = print_ma(A, B, DIM, K)
        self.assertEqual(
            strings,
            [
                "w = 2.71828 * (u_1)**2 * (u_2)**3 * (u_3)**4",
                "w = 148.413 * (u_1)**6 * (u_2)**7 * (u_3)**8",
                "w = 8103.08 * (u_1)**10 * (u_2)**11 * (u_3)**12",
            ],
        )

    def test_sma(self):
        strings = print_sma(A, B, ALPHA[0], DIM, K)
        self.assertEqual(
            strings,
            [
                "w**0.0769231 = 1.07996 * (u_1)**0.153846 "
                "* (u_2)**0.230769 * (u_3)**0.307692",
                "    + 1.46905 * (u_1)**0.461538 * "
                "(u_2)**0.538462 * (u_3)**0.615385",
                "    + 1.99832 * (u_1)**0.769231 * "
                "(u_2)**0.846154 * (u_3)**0.923077",
            ],
        )

    def test_isma(self):
        strings = print_isma(A, B, ALPHA, DIM, K)
        self.assertEqual(
            strings,
            [
                "1 = (1.07996/w**0.0769231) * (u_1)**0.153846 * "
                "(u_2)**0.230769 * (u_3)**0.307692",
                "    + (1.42924/w**0.0714286) * (u_1)**0.428571 * "
                "(u_2)**0.5 * (u_3)**0.571429",
                "    + (1.82212/w**0.0666667) * (u_1)**0.666667 * "
                "(u_2)**0.733333 * (u_3)**0.8",
            ],
        )


TESTS = [TestPrintFit]

if __name__ == "__main__":
    SUITE = unittest.TestSuite()
    LOADER = unittest.TestLoader()

    for t in TESTS:
        SUITE.addTests(LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(SUITE)
