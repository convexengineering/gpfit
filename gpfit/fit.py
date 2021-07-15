"Implements the all-important 'fit' function."
from numpy import ones, exp, sqrt, mean, square, hstack
from .classes import max_affine, softmax_affine, implicit_softmax_affine
from .least_squares import levenberg_marquardt
from .initialize import get_initial_parameters
from .print_fit import print_isma, print_sma, print_ma
from .constraint_set import FitConstraintSet

CLASSES = {
    "ISMA": implicit_softmax_affine,
    "SMA": softmax_affine,
    "MA": max_affine,
}


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=import-error
def fit(xdata, ydata, K, ftype="ISMA", alpha0=10):
    """
    Fit a log-convex function to multivariate data, returning a FitConstraintSet

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
        FitConstraintSet

        rms_error:  float
            Root mean square error between original (not log transformed)
            data and function fit.
    """

    if ydata.ndim > 1:
        raise ValueError("Dependent data should be a 1D numpy array")

    xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T
    d = int(xdata.shape[1])  # Number of dimensions
    fitdata = {"ftype": ftype, "K": K, "d": d}
    if d == 1:
        fitdata["lb0"] = exp(min(xdata.reshape(xdata.size)))
        fitdata["ub0"] = exp(max(xdata.reshape(xdata.size)))
    else:
        for i in range(d):
            fitdata["lb%d" % i] = exp(min(xdata.T[i]))
            fitdata["ub%d" % i] = exp(max(xdata.T[i]))

    def residual(params):
        "A specific residual function."
        [yhat, drdp] = CLASSES[ftype](xdata, params)
        r = yhat - ydata
        return r, drdp

    ba = get_initial_parameters(xdata, ydata.reshape(ydata.size, 1),
                                K).flatten("F")
    if ftype == "ISMA":
        params, _ = levenberg_marquardt(residual, hstack((ba, alpha0*ones(K))))
    elif ftype == "SMA":
        params, _ = levenberg_marquardt(residual, hstack((ba, alpha0)))
    else:
        params, _ = levenberg_marquardt(residual, ba)

    # A: exponent parameters, B: coefficient parameters
    A = params[[i for i in range(K*(d + 1)) if i % (d + 1) != 0]]
    B = params[[i for i in range(K*(d + 1)) if i % (d + 1) == 0]]

    if ftype == "ISMA":
        alpha = 1/params[list(range(-K, 0))]
        for k in range(K):
            fitdata["c%d" % k] = exp(alpha[k]*B[k])
            fitdata["a%d" % k] = alpha[k]
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = alpha[k]*A[d*k + i]
        print_isma(A, B, alpha, d, K)
    elif ftype == "SMA":
        alpha = 1/params[-1]
        fitdata["a1"] = alpha
        for k in range(K):
            fitdata["c%d" % k] = exp(alpha*B[k])
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = alpha*A[d*k + i]
        print_sma(A, B, alpha, d, K)
    elif ftype == "MA":
        alpha = 1
        fitdata["a1"] = 1
        for k in range(K):
            fitdata["c%d" % k] = exp(B[k])
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = A[d*k + i]
        print_ma(A, B, d, K)

    if min(exp(B*alpha)) < 1e-100:
        raise ValueError("Fitted constraint contains too small a value...")
    if max(exp(B*alpha)) > 1e100:
        raise ValueError("Fitted constraint contains too large a value...")

    def evaluate(xdata):
        """Evaluate the fit over a range of xdata."""
        xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T
        return CLASSES[ftype](xdata, params)[0]

    fitdata["rms_err"] = sqrt(mean(square(evaluate(xdata.T) - ydata)))
    fitdata["max_err"] = sqrt(max(square(evaluate(xdata.T) - ydata)))

    cs = FitConstraintSet(fitdata)
    cs.evaluate = evaluate

    return cs, cs.rms_err
