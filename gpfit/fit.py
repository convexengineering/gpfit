"Implements the all-important 'fit' function."
from numpy import ones, exp, sqrt, mean, square, hstack
from .classes import max_affine, softmax_affine, implicit_softmax_affine
from .levenberg_marquardt import levenberg_marquardt
from .initialize import get_initial_parameters
from .print_fit import print_isma, print_sma, print_ma
from .constraint_set import FitConstraintSet

ALPHA0 = 10
CLASSES = {
    "ISMA": implicit_softmax_affine,
    "SMA": softmax_affine,
    "MA": max_affine,
}


# pylint: disable=invalid-name
def get_parameters(ftype, K, xdata, ydata):
    "Perform least-squares fitting optimization."

    ydata_col = ydata.reshape(ydata.size, 1)
    ba = get_initial_parameters(xdata, ydata_col, K).flatten('F')

    def residual(params):
        "A specific residual function."
        [yhat, drdp] = CLASSES[ftype](xdata, params)
        r = yhat - ydata
        return r, drdp

    if ftype == "ISMA":
        params, _ = levenberg_marquardt(residual, hstack((ba, ALPHA0*ones(K))))
    elif ftype == "SMA":
        params, _ = levenberg_marquardt(residual, hstack((ba, ALPHA0)))
    else:
        params, _ = levenberg_marquardt(residual, ba)

    return params


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=import-error
def fit(xdata, ydata, K, ftype="ISMA"):
    """
    Fit a log-convex function to multivariate data, returning a FitConstraintSet

    INPUTS
        xdata:      Independent variable data
                        2D numpy array [nDim, nPoints]
                        [[<--------- x1 ------------->]
                         [<--------- x2 ------------->]]

        ydata:      Dependent variable data
                        1D numpy array [nPoints,]
                        [<---------- y ------------->]

        rms_error:  float
            Root mean square error between original (not log transformed)
            data and function fit.
    """

    # Check data is in correct form
    if ydata.ndim > 1:
        raise ValueError('Dependent data should be a 1D numpy array')

    # Transform to column vector
    xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T

    # Dimension of function (number of independent variables)
    d = int(xdata.shape[1])

    # begin fit data dict to generate fit constraint set
    fitdata = {"ftype": ftype, "K": K, "d": d}

    # find bounds of x
    if d == 1:
        fitdata["lb0"] = exp(min(xdata.reshape(xdata.size)))
        fitdata["ub0"] = exp(max(xdata.reshape(xdata.size)))
    else:
        for i in range(d):
            fitdata["lb%d" % i] = exp(min(xdata.T[i]))
            fitdata["ub%d" % i] = exp(max(xdata.T[i]))

    params = get_parameters(ftype, K, xdata, ydata)

    # A: exponent parameters, B: coefficient parameters
    A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
    B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

    if ftype == "ISMA":
        alpha = 1./params[list(range(-K, 0))]
        for k in range(K):
            fitdata["c%d" % k] = exp(alpha[k]*B[k])
            fitdata["a%d" % k] = alpha[k]
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = alpha[k]*A[d*k+i]
        print_isma(A, B, alpha, d, K)
    elif ftype == "SMA":
        alpha = 1./params[-1]
        fitdata["a1"] = alpha
        for k in range(K):
            fitdata["c%d" % k] = exp(alpha*B[k])
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = alpha*A[d*k+i]
        print_sma(A, B, alpha, d, K)
    elif ftype == "MA":
        alpha = 1
        fitdata["a1"] = 1
        for k in range(K):
            fitdata["c%d" % k] = exp(B[k])
            for i in range(d):
                fitdata["e%d%d" % (k, i)] = A[d*k+i]
        print_ma(A, B, d, K)

    if min(exp(B*alpha)) < 1e-100:
        raise ValueError("Fitted constraint contains too small a value...")
    if max(exp(B*alpha)) > 1e100:
        raise ValueError("Fitted constraint contains too large a value...")

    def evaluate(xdata):
        """
        Evaluate the y of this fit over a range of xdata.

        INPUTS
            xdata:      Independent variable data
                            2D numpy array [nDim, nPoints]
                            [[<--------- x1 ------------->]
                             [<--------- x2 ------------->]]

        OUTPUT
            ydata:      Dependent variable data in 1D numpy array
        """
        xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T
        return CLASSES[ftype](xdata, params)[0]

    # cstrt.evaluate = evaluate
    fitdata["rms_err"] = sqrt(mean(square(evaluate(xdata.T)-ydata)))
    fitdata["max_err"] = sqrt(max(square(evaluate(xdata.T)-ydata)))

    cs = FitConstraintSet(fitdata)
    cs.evaluate = evaluate

    return cs, cs.rms_err
