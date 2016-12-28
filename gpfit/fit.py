"Implements the all-important 'fit' function."
from numpy import append, ones, exp, sqrt, mean, square
from gpkit import NamedVariables, VectorVariable, Variable, NomialArray
from .implicit_softmax_affine import implicit_softmax_affine
from .softmax_affine import softmax_affine
from .max_affine import max_affine
from .LM import LM
from .generic_resid_fun import generic_resid_fun
from .max_affine_init import max_affine_init
from .print_fit import print_ISMA, print_SMA, print_MA


RFUN = {"ISMA": implicit_softmax_affine,
        "SMA": softmax_affine,
        "MA": max_affine}


def _fit(ftype, K, xdata, ydata):
    "Perform least-squares fitting optimization."
    def rfun(p):
        "A specific residual function."
        return generic_resid_fun(RFUN[ftype], xdata, ydata, p)

    alphainit = 10
    bainit = max_affine_init(xdata, ydata, K).flatten('F')

    if ftype == "ISMA":
        params, _ = LM(rfun, append(bainit, alphainit*ones((K, 1))))
    elif ftype == "SMA":
        params, _ = LM(rfun, append(bainit, alphainit))
    else:
        params, _ = LM(rfun, bainit)

    y_fit, _ = RFUN[ftype](xdata, params)
    rms_error = sqrt(mean(square(y_fit-ydata.T[0])))

    return params, rms_error


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

        K:          Number of terms in the fit
                        integer > 0

        ftype:      Fit type
            one of the following strings: "ISMA" (default), "SMA" or "MA"

    OUTPUTS
        cstrt:      GPkit constraint
            For K > 1, this will be a posynomial inequality or set thereof
            If K = 1, this is automatically made into an equality constraint

        rms_error:  float
            Root mean square error between original (not log transformed)
            data and function fit.
    """

    # Check data is in correct form
    if ydata.ndim > 1:
        raise ValueError('Dependent data should be in the form of a 1D numpy array')

    # Transform to column vector
    ydata = ydata.reshape(ydata.size, 1)
    xdata = xdata.reshape(xdata.size, 1) if xdata.ndim == 1 else xdata.T

    # Dimension of function (number of independent variables)
    d = int(xdata.shape[1])

    with NamedVariables("fit"):
        u = VectorVariable(d, "u")
        w = Variable("w")

    params, rms_error = _fit(ftype, K, xdata, ydata)

    # A: exponent parameters, B: coefficient parameters
    A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
    B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

    if ftype == "ISMA":
        alpha = 1./params[range(-K, 0)]
    elif ftype == "SMA":
        alpha = 1./params[-1]
    elif ftype == "MA":
        alpha = 1

    monos = NomialArray([e_b * (u**A[k*d:(k+1)*d]).prod()
                         for k, e_b in enumerate(exp(B))])**alpha

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

    return cstrt, rms_error
