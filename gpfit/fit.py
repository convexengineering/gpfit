"""Implements the Fit class"""
import numpy as np
import matplotlib.pyplot as plt
from .least_squares import levenberg_marquardt
from .initialize import get_initial_parameters
from .logsumexp import lse_scaled, lse_implicit


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=import-error
# pylint: disable=too-many-instance-attributes
class Fit:
    """The base class for GPfit"""
    def __init__(self, xdata, ydata, K, alpha0=10, verbosity=1):
        """
        Initialize Fit object

        Arguments
        ---------
            xdata:      Independent variable data
                            2D numpy array [nDim, nPoints]
                            [[<--------- x1 ------------->]
                             [<--------- x2 ------------->]]

            ydata:      Dependent variable data
                            1D numpy array [nPoints,]
                            [<---------- y ------------->]

            K:          Number of terms

            ftype:      Function class ["MA", "SMA", "ISMA"]

            alpha0:     Initial guess for smoothing parameter alpha

        Returns
        -------
        """

        if ydata.ndim > 1:
            raise ValueError("Dependent data should be a 1D numpy array")

        self.ydata = ydata
        self.xdata = xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T
        self.d = d = int(xdata.shape[1])  # Number of dimensions
        self.K = K
        self.fitdata = {"K": K, "d": d, "alpha0": alpha0}
        if d == 1:
            self.fitdata["lb0"] = np.exp(min(xdata.reshape(xdata.size)))
            self.fitdata["ub0"] = np.exp(max(xdata.reshape(xdata.size)))
        else:
            for i in range(d):
                self.fitdata["lb%d" % i] = np.exp(min(xdata.T[i]))
                self.fitdata["ub%d" % i] = np.exp(max(xdata.T[i]))

        ba = get_initial_parameters(xdata, ydata.reshape(self.ydata.size, 1),
                                    K).flatten("F")
        self.A, self.B, self.alpha, self.params = self.get_parameters(ba, K, d)

        self.error = {
            "rms": np.sqrt(np.mean(np.square(self.evaluate(xdata, self.params)[0] - ydata))),
            "max": np.sqrt(max(np.square(self.evaluate(xdata, self.params)[0] - ydata)))
            # TODO: check max, check .T behaviour of old code
        }

        if verbosity >= 1:
            self.print_fit()
            print(self.error["rms"])

    def residual(self, params):
        """Calculate residual"""
        [yhat, drdp] = self.evaluate(self.xdata, params)
        r = yhat - self.ydata
        return r, drdp

    def plot_fit(self):
        """Plots fit"""
        f, ax = plt.subplots()
        udata = np.exp(self.xdata)
        wdata = np.exp(self.ydata)
        ax.plot(udata, wdata, "+r")

        xx = np.linspace(min(self.xdata), max(self.xdata), 10)
        yy, _ = self.evaluate(xx, self.params)
        uu = np.exp(xx)
        ww = np.exp(yy)
        ax.plot(uu, ww)

        stringlist = self.print_fit()
        ax.set_xlabel("u")
        ax.set_ylabel("w")
        ax.legend(["Data"] + stringlist, loc="best")
        return f, ax


class MaxAffine(Fit):
    """Max Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        params, _ = levenberg_marquardt(self.residual, ba)

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1

        self.fitdata["a1"] = alpha
        for k in range(K):
            self.fitdata["c%d" % k] = np.exp(B[k])
            for i in range(d):
                self.fitdata["e%d%d" % (k, i)] = A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates max affine function at values of x, given a set of
        max affine fit parameters.

        Arguments
        ---------
            x: 2D array [nPoints x nDim]
                Independent variable data

            ba: 2D array
                max affine fit parameters
                [[b1, a11, ... a1k]
                 [ ....,          ]
                 [bk, ak1, ... akk]]

        Returns
        -------
            y: 1D array [nPoints]
                Max affine output
            dydba: 2D array [nPoints x (nDim + 1)*K]
                dydba
        """
        ba = params
        npt, dimx = x.shape
        K = ba.size // (dimx + 1)
        ba = np.reshape(ba, (dimx + 1, K), order="F")  # 'F' gives Fortran indexing
        X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
        y, partition = np.dot(X, ba).max(1), np.dot(X, ba).argmax(1)

        dydba = np.zeros((npt, (dimx + 1)*K))
        for k in range(K):
            inds = np.equal(partition, k)
            indadd = (dimx + 1)*k
            ixgrid = np.ix_(inds.nonzero()[0], indadd + np.arange(dimx + 1))
            dydba[ixgrid] = X[inds, :]

        return y, dydba

    def print_fit(self):
        """Print fit"""
        K, d = self.K, self.d
        A, B = self.A, self.B
        string_list = [None]*K
        for k in range(K):
            print_string = "w = {0:.6g}".format(np.exp(B[k]))
            for i in range(d):
                print_string += " * (u_{0:d})**{1:.6g}".format(i + 1, A[d*k + i])
            string_list[k] = print_string
            print(print_string)
        return string_list


class SoftmaxAffine(Fit):
    """Softmax Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        alpha0 = self.fitdata["alpha0"]
        params, _ = levenberg_marquardt(self.residual, np.hstack((ba, alpha0)))

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1/params[-1]

        self.fitdata["a1"] = alpha
        for k in range(K):
            self.fitdata["c%d" % k] = np.exp(alpha*B[k])
            for i in range(d):
                self.fitdata["e%d%d" % (k, i)] = alpha*A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates softmax affine function at values of x, given a set of
        SMA fit parameters.

        Arguments:
        ----------
                x:      Independent variable data
                            2D numpy array [nPoints x nDimensions]

                params: Fit parameters
                            1D numpy array [(nDim + 2)*K,]
                            [b1, a11, .. a1d, b2, a21, .. a2d, ...
                             bK, aK1, aK2, .. aKd, alpha]

        Returns:
        --------
                y:      SMA approximation to log transformed data
                            1D numpy array [nPoints]

                dydp:   Jacobian matrix
        """

        npt, dimx = x.shape
        ba = params[0:-1]
        softness = params[-1]
        alpha = 1/softness
        if alpha <= 0:
            return np.inf*np.ones((npt, 1)), np.nan
        K = np.size(ba) // (dimx + 1)
        ba = ba.reshape(dimx + 1, K, order="F")
        X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
        z = np.dot(X, ba)  # compute affine functions
        y, dydz, dydsoftness = lse_scaled(z, alpha)

        dydsoftness = -dydsoftness*(alpha**2)
        nrow, ncol = dydz.shape
        repmat = np.tile(dydz, (dimx + 1, 1)).reshape(nrow, ncol*(dimx + 1), order="F")
        dydba = repmat*np.tile(X, (1, K))
        dydsoftness.shape = (dydsoftness.size, 1)
        dydp = np.hstack((dydba, dydsoftness))

        return y, dydp

    def print_fit(self):
        """Print fit"""
        K, d = self.K, self.d
        A, B, alpha = self.A, self.B, self.alpha
        string_list = [None]*K
        print_string = "w**{0:.6g} = ".format(alpha)
        for k in range(K):
            if k > 0:
                print(print_string)
                print_string = "    + "
            print_string += "{0:.6g}".format(np.exp(alpha*B[k]))
            for i in range(d):
                print_string += " * (u_{0:d})**{1:.6g}".format(i + 1, alpha*A[d*k + i])
            string_list[k] = print_string
        print(print_string)
        return ["".join(string_list)]


class ImplicitSoftmaxAffine(Fit):
    """Implicit Softmax Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        alpha0 = self.fitdata["alpha0"]
        params, _ = levenberg_marquardt(self.residual, np.hstack((ba, alpha0*np.ones(K))))

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1/params[list(range(-K, 0))]

        for k in range(K):
            self.fitdata["c%d" % k] = np.exp(alpha[k]*B[k])
            self.fitdata["a%d" % k] = alpha[k]
            for i in range(d):
                self.fitdata["e%d%d" % (k, i)] = alpha[k]*A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates implicit softmax affine function at values of x, given a set of
        ISMA fit parameters.

        Arguments:
        ----------
                x:      Independent variable data
                            2D numpy array [nPoints x nDimensions]

                params: Fit parameters
                            1D numpy array [(nDim + 2)*K,]
                            [b1, a11, .. a1d, b2, a21, .. a2d, ...
                             bK, aK1, aK2, .. aKd, alpha1, alpha2, ... alphaK]

        Returns:
        --------
                y:      ISMA approximation to log transformed data
                            1D numpy array [nPoints]

                dydp:   Jacobian matrix

        """

        npt, dimx = x.shape
        K = params.size // (dimx + 2)
        ba = params[0:-K]
        alpha = params[-K:]
        if any(alpha <= 0):
            return np.inf*np.ones((npt, 1)), np.nan
        ba = ba.reshape(dimx + 1, K, order="F")  # reshape ba to matrix
        X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
        z = np.dot(X, ba)  # compute affine functions
        y, dydz, dydalpha = lse_implicit(z, alpha)

        nrow, ncol = dydz.shape
        repmat = np.tile(dydz, (dimx + 1, 1)).reshape(nrow, ncol*(dimx + 1), order="F")
        dydba = repmat*np.tile(X, (1, K))
        dydp = np.hstack((dydba, dydalpha))

        return y, dydp

    def print_fit(self):
        """Print fit"""
        K, d = self.K, self.d
        A, B, alpha = self.A, self.B, self.alpha
        string_list = [None]*K
        print_string = "1 = "
        for k in range(K):
            if k > 0:
                print(print_string)
                print_string = "    + "
            print_string += "({0:.6g}/w**{1:.6g})".format(np.exp(alpha[k]*B[k]), alpha[k])
            for i in range(d):
                print_string += " * (u_{0:d})**{1:.6g}".format(
                    i + 1, alpha[k]*A[d*k + i]
                )
            string_list[k] = print_string
        print(print_string)
        return ["".join(string_list)]
