"Implements the all-important 'fit' function."
from numpy import ones, exp, sqrt, mean, square, hstack
from gpkit import NamedVariables, VectorVariable, Variable, NomialArray
from .implicit_softmax_affine import implicit_softmax_affine
from .softmax_affine import softmax_affine
from .max_affine import max_affine
from .levenberg_marquardt import levenberg_marquardt
from .ba_init import ba_init
from .print_fit import print_ISMA, print_SMA, print_MA
import pandas as pd

ALPHA_INIT = 10
RFUN = {"ISMA": implicit_softmax_affine,
        "SMA": softmax_affine,
        "MA": max_affine}


def get_params(ftype, K, xdata, ydata):
    "Perform least-squares fitting optimization."
    def rfun(params):
        "A specific residual function."
        [yhat, drdp] = RFUN[ftype](xdata, params)
        r = yhat - ydata
        return r, drdp

    ba = ba_init(xdata, ydata.reshape(ydata.size, 1), K).flatten('F')

    if ftype == "ISMA":
        params, _ = levenberg_marquardt(rfun, hstack((ba, ALPHA_INIT*ones(K))))
    elif ftype == "SMA":
        params, _ = levenberg_marquardt(rfun, hstack((ba, ALPHA_INIT)))
    else:
        params, _ = levenberg_marquardt(rfun, ba)

    return params

def get_dataframe(ftype, K, xdata, ydata):

    # Transform to column vector
    xdatat = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T

    # Dimension of function (number of independent variables)
    d = int(xdatat.shape[1])
    bounds = []
    for i in range(d):
        bounds.append(min(xdata[i]))
        bounds.append(max(xdata[i]))

    params = get_params(ftype, K, xdatat, ydata)

    # A: exponent parameters, B: coefficient parameters
    A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
    B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

    cs = []
    exs = []
    if ftype == "ISMA":
        alpha = 1./params[range(-K, 0)]
        for k in range(K):
            cs.append(exp(alpha[k]*B[K]))
            for i in range(d):
                exs.append(alpha[k]*A[d*k + i])
    elif ftype == "SMA":
        alpha = 1./params[-1]
        for k in range(K):
            cs.append(exp(alpha*B[k]))
            for i in range(d):
                exs.append(alpha*A[d*k + i])
        alpha = [alpha]
    elif ftype == "MA":
        alpha = 1
        for k in range(K):
            cs.append(exp(B[k]))
            for i in range(d):
                exs.append(A[d*k + i])
        alpha = [alpha]

    data = hstack([cs, exs, alpha, exp(bounds)])
    df = pd.DataFrame(data).transpose()
    colnames = hstack(["c%d" % k for k in range(1, K+1)])
    colnames = hstack([colnames, ["e%d%d" % (k, i) for k in range(1, K+1)
                                  for i in range(1, d+1)]])
    colnames = hstack([colnames, ["a%d" % i for i in range(1, len(alpha)+1)]])
    colnames = hstack([colnames, hstack([["lb%d" % i, "ub%d" % i]
                                         for i in range(1, d+1)])])
    df.columns = colnames
    return df

# pylint: disable=too-many-locals
def fit(xdata, ydata, K, ftype="ISMA"):
    """
    Fit a log-convex function to multivariate data, returning a GP constraint

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
        raise ValueError('Dependent data should be in the form of a 1D numpy array')

    # Transform to column vector
    xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T

    # Dimension of function (number of independent variables)
    d = int(xdata.shape[1])

    with NamedVariables("fit"):
        u = VectorVariable(d, "u")
        w = Variable("w")

    params = get_params(ftype, K, xdata, ydata)

    # A: exponent parameters, B: coefficient parameters
    A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
    B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

    if ftype == "ISMA":
        alpha = 1./params[range(-K, 0)]
    elif ftype == "SMA":
        alpha = 1./params[-1]
    elif ftype == "MA":
        alpha = 1

    monos = exp(B*alpha) * NomialArray([(u**A[k*d:(k+1)*d]).prod()
                                        for k in range(K)])**alpha

    if ftype == "ISMA":
        # constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
        lhs, rhs = 1, (monos/w**alpha).sum()
        print_ISMA(A, B, alpha, d, K)
    elif ftype == "SMA":
        # constraint of the form w^alpha >= c1*u1^exp1 + c2*u2^exp2 +....
        lhs, rhs = w**alpha, monos.sum()
        print_SMA(A, B, alpha, d, K)
    elif ftype == "MA":
        # constraint of the form w >= c1*u1^exp1, w >= c2*u2^exp2, ....
        lhs, rhs = w, monos
        print_MA(A, B, d, K)

    if K == 1:
        # when possible, return an equality constraint
        cstrt = (lhs == rhs)
    else:
        cstrt = (lhs >= rhs)

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
        return RFUN[ftype](xdata, params)[0]

    cstrt.evaluate = evaluate
    rms_error = sqrt(mean(square(evaluate(xdata.T)-ydata)))
    max_error = sqrt(max(square(evaluate(xdata.T)-ydata)))

    return cstrt, [rms_error, max_error]
