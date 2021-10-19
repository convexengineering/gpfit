"""The fit classes (both in the python sense and the mathematical sense)"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from .maths.least_squares import levenberg_marquardt
from .maths.initialize import get_initial_parameters
from .maths.logsumexp import lse_scaled, lse_implicit
from .constraint_set import FitConstraintSet


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class _Fit:
    """The base class for GPfit"""
    def __init__(self, xdata, ydata, K, alpha0=10, verbosity=0, seed=None):
        """Initialize _Fit object

        Arguments
        ---------
        xdata: 2D numpy array [nDim, nPoints]
            Independent variable data

        ydata: 1D numpy array [nPoints,]
            Dependent variable data

        K: int
            Number of terms

        alpha0:
            Initial guess for smoothing parameter alpha

        verbosity: int (0 or 1)
            Verbosity

        seed: None or int
            Seed for random number generator in initialization function

        """

        if ydata.ndim > 1:
            raise ValueError("Dependent data should be a 1D numpy array")

        self.ydata = ydata
        self.xdata = xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T
        self.d = d = int(xdata.shape[1])  # Number of dimensions
        self.K = K
        self.type = type(self).__name__
        self.parameters = {"alpha0": alpha0}
        self.bounds = {}
        if d == 1:
            self.bounds["lb0"] = np.exp(min(xdata.reshape(xdata.size)))
            self.bounds["ub0"] = np.exp(max(xdata.reshape(xdata.size)))
        else:
            for i in range(d):
                self.bounds[f"lb{i}"] = np.exp(min(xdata.T[i]))
                self.bounds[f"ub{i}"] = np.exp(max(xdata.T[i]))

        ba = get_initial_parameters(xdata, ydata.reshape(self.ydata.size, 1),
                                    K, seed).flatten("F")
        self.A, self.B, self.alpha, self.params = self.get_parameters(ba, K, d)

        yhat = self.evaluate(xdata, self.params)[0]
        yerror = yhat - ydata
        what = np.exp(yhat)
        wdata = np.exp(self.ydata)
        werror = what - wdata
        self.errors = {
            "rms_log": np.sqrt(np.mean(np.square(yerror))),
            "rms_abs": np.sqrt(np.mean(np.square(werror))),
            "rms_rel": np.sqrt(np.mean(np.square(werror/wdata))),
            "max_abs": max(abs(werror)),
            "max_rel": max(abs(werror/wdata)),
        }
        self.error = self.errors["rms_rel"]

        if verbosity >= 1:
            self.print_result()

    def residual(self, params):
        """Calculate residual"""
        [yhat, drdp] = self.evaluate(self.xdata, params)
        r = yhat - self.ydata
        return r, drdp

    def plot(self):
        """Plots fit alongside original data for a 1D fit"""
        f, ax = plt.subplots()
        udata = np.exp(self.xdata)
        wdata = np.exp(self.ydata)
        ax.plot(udata, wdata, "+r")

        xx = np.linspace(min(self.xdata), max(self.xdata), 10)
        yy, _ = self.evaluate(xx, self.params)
        uu = np.exp(xx)
        ww = np.exp(yy)
        ax.plot(uu, ww)

        stringlist = self.__repr__()
        ax.set_xlabel("u")
        ax.set_ylabel("w")
        ax.legend(["Data"] + [stringlist], loc="best")
        return f, ax

    def plot_surface(self, azim=0):
        """Plots surface of fit alongside original data"""
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_zlabel("w")
        ax.azim = azim
        # Plot original data
        udata = np.exp(self.xdata)
        wdata = np.exp(self.ydata)
        ax.scatter3D(udata[:, 0], udata[:, 1], wdata, color="r")
        # Plot surface of fit
        x1 = np.linspace(min(self.xdata[:, 0]), max(self.xdata[:, 0]), 10)
        x2 = np.linspace(min(self.xdata[:, 1]), max(self.xdata[:, 1]), 10)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx = np.vstack((xx1.flatten(), xx2.flatten()))
        yy, _ = self.evaluate(xx.T, self.params)
        uu1, uu2 = np.exp(xx1), np.exp(xx2)
        ww = np.exp(yy)
        ax.plot_surface(uu1, uu2, ww.reshape(uu1.shape), cmap=cm.coolwarm,
                        alpha=0.8)
        return fig, ax

    def plot_slices(self):
        """
        Plots slices of fit alongside original data.

        x-axis is first dependent variable, each slice is at a different value
        of the second dependent variable.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("u1")
        ax.set_ylabel("w")
        # Plot original data
        udata = np.exp(self.xdata)
        wdata = np.exp(self.ydata)
        norm = Normalize(vmin=min(udata[:, 1]), vmax=max(udata[:, 1]))
        ax.scatter(udata[:, 0], wdata, c=cm.viridis(norm(udata[:, 1])))
        # Plot slices
        x1 = np.linspace(min(self.xdata[:, 0]), max(self.xdata[:, 0]), 10)
        x2slices = np.unique(self.xdata[:, 1])
        for x2slice in x2slices:
            x2 = x2slice*np.ones(x1.shape)
            xx = np.vstack((x1, x2))
            yy, _ = self.evaluate(xx.T, self.params)
            u1, u2slice = np.exp(x1), np.exp(x2slice)
            ww = np.exp(yy)
            ax.plot(u1, ww, c=cm.viridis(norm(u2slice)),
                    label=f"{u2slice:.3g}")
            ax.legend(title="u2")
        return fig, ax

    def save(self, filename="fit.pkl"):
        """Save Fit object to pickle"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def savetxt(self, filename="fit.txt"):
        """Save string of fit to text file using python math convention (**)"""
        fitstr = self.__repr__().replace("^", "**")
        with open(filename, "w", encoding="utf-8") as textfile:
            textfile.write(fitstr)

    def constraint_set(self, **kwargs):
        """Returns constraint set"""
        return FitConstraintSet(self, **kwargs)

    def print_result(self):
        """Prints time, fit, and error"""

        fitstr = self.__repr__()

        printstr = (
            f"Generated {self.type} fit with {self.K} terms.\n\n"
            f"Fit\n"
            f"---\n"
            f"{fitstr}\n\n"
            f"Error\n"
            f"-----\n"
            f"""RMS: {self.errors["rms_rel"]*100:.2g}%\n"""
            f"""Max: {self.errors["max_rel"]*100:.2g}%\n"""
        )
        print(printstr)


class MaxAffine(_Fit):
    """Max Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        params, _ = levenberg_marquardt(self.residual, ba)

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1

        self.parameters["a1"] = alpha
        for k in range(K):
            self.parameters[f"c{k}"] = np.exp(B[k])
            for i in range(d):
                self.parameters[f"e{k}{i}"] = A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates max affine function at values of x, given a set of
        max affine fit parameters.

        Args
        ----
        x: 2D array [nPoints x nDim]
            Independent variable data

        ba: 2D array
            Max affine fit parameters
            [[b1, a11, ... a1k], [...], [bk, ak1, ... akk]]

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

    def __repr__(self):
        """String representation of fit"""
        K, d = self.K, self.d
        A, B = self.A, self.B
        string_list = [None]*K
        for k in range(K):
            print_string = f"w = {np.exp(B[k]):.6g}"
            for i in range(d):
                print_string += f"*(u_{i + 1})^{A[d*k + i]:.6g}"
            string_list[k] = print_string
        return "\n".join(string_list)


class SoftmaxAffine(_Fit):
    """Softmax Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        alpha0 = self.parameters["alpha0"]
        params, _ = levenberg_marquardt(self.residual, np.hstack((ba, alpha0)))

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1/params[-1]

        self.parameters["a1"] = alpha
        for k in range(K):
            self.parameters[f"c{k}"] = np.exp(alpha*B[k])
            for i in range(d):
                self.parameters[f"e{k}{i}"] = alpha*A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates softmax affine function at values of x, given a set of
        SMA fit parameters.

        Arguments:
        ----------
        x: 2D numpy array [nPoints x nDimensions]
            Independent variable data

        params: 1D numpy array [(nDim + 2)*K,]
            Fit parameters
            [b1, a11, .. a1d, b2, a21, .. a2d, ... bK, aK1, aK2, .. aKd, alpha]

        Returns:
        --------
        y: 1D numpy array [nPoints]
            SMA approximation to log transformed data

        dydp
            Jacobian matrix
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

    def __repr__(self):
        """String representation of fit"""
        K, d = self.K, self.d
        A, B, alpha = self.A, self.B, self.alpha
        string_list = [None]*K
        print_string = f"w^{alpha:.6g} = "
        for k in range(K):
            if k > 0:
                print_string = "    + "
            print_string += f"{np.exp(alpha*B[k]):.6g}"
            for i in range(d):
                print_string += f"*(u_{i + 1:d})^{alpha*A[d*k + i]:.6g}"
            string_list[k] = print_string
        return "\n".join(string_list)


class ImplicitSoftmaxAffine(_Fit):
    """Implicit Softmax Affine fit class"""

    def get_parameters(self, ba, K, d):
        """Get fit parameters"""
        alpha0 = self.parameters["alpha0"]
        params, _ = levenberg_marquardt(self.residual, np.hstack((ba, alpha0*np.ones(K))))

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]
        alpha = 1/params[list(range(-K, 0))]

        for k in range(K):
            self.parameters[f"c{k}"] = np.exp(alpha[k]*B[k])
            self.parameters[f"a{k}"] = alpha[k]
            for i in range(d):
                self.parameters[f"e{k}{i}"] = alpha[k]*A[d*k + i]
        return A, B, alpha, params

    @staticmethod
    def evaluate(x, params):
        """
        Evaluates implicit softmax affine function at values of x, given a set
        of ISMA fit parameters.

        Arguments:
        ----------
        x: 2D numpy array [nPoints x nDimensions]
            Independent variable data

        params: 1D numpy array [(nDim + 2)*K,]
            Fit parameters
            [b1, a11, .. a1d, b2, a21, .. a2d, ..., bK, aK1, aK2, .. aKd,
            alpha1, alpha2, ... alphaK]

        Returns:
        --------
        y: 1D numpy array [nPoints]
            ISMA approximation to log transformed data

        dydp
            Jacobian matrix

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

    def __repr__(self):
        """String representation of fit"""
        K, d = self.K, self.d
        A, B, alpha = self.A, self.B, self.alpha
        string_list = [None]*K
        print_string = "1 = "
        for k in range(K):
            if k > 0:
                print_string = "  + "
            print_string += f"({np.exp(alpha[k]*B[k]):.6g}/w^{alpha[k]:.6g})"
            for i in range(d):
                print_string += f"*(u_{i + 1:d})^{alpha[k]*A[d*k + i]:.6g}"
            string_list[k] = print_string
        return "\n".join(string_list)
