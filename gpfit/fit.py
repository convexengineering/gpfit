"Implements the all-important 'fit' function."
from numpy import ones, exp, sqrt, mean, square, hstack, array
from gpkit import NamedVariables, VectorVariable, Variable, NomialArray
from .implicit_softmax_affine import implicit_softmax_affine
from .softmax_affine import softmax_affine
from .max_affine import max_affine
from .levenberg_marquardt import levenberg_marquardt
from .ba_init import ba_init
from .print_fit import print_ISMA, print_SMA, print_MA

ALPHA_INIT = 10
RFUN = {"ISMA": implicit_softmax_affine,
        "SMA": softmax_affine,
        "MA": max_affine}


# pylint: disable=invalid-name
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


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
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
        raise ValueError('Dependent data should be a 1D numpy array')

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
        cs = rhs.cs
        exps = hstack([[exs[e] for e in exs if "w" not in str(e)]
                       for exs in list(rhs.exps)])
        print_ISMA(A, B, alpha, d, K)
    elif ftype == "SMA":
        # constraint of the form w^alpha >= c1*u1^exp1 + c2*u2^exp2 +....
        lhs, rhs = w**alpha, monos.sum()
        cs = rhs.cs
        exps = hstack([[exs[e] for e in exs] for exs in list(rhs.exps)])
        print_SMA(A, B, alpha, d, K)
    elif ftype == "MA":
        # constraint of the form w >= c1*u1^exp1, w >= c2*u2^exp2, ....
        lhs, rhs = w, monos
        cs = hstack([fn.cs for fn in rhs])
        exps = hstack([[fn.exps[0][e] for e in fn.exps[0]] for fn in rhs])
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

    def get_dataframe(xdata):
        """
        Returns fit parameters as a dataframe
        -------------------------------------
        INPUTS
            xdata:      Independent variable data
                            2D numpy array [nDim, nPoints]
            ** Must have pandas installed

        OUTPUT
            df:         Fitted constraint parameters
                            Pandas dataframe
                            ex:
                w**a1 = c1 * u_1**e11 * u_2**e12 + c2 * u_1**e21 * u_2**e22
                df.columns = ["ftype", "K", "d", "c1", "c2", "e11", "e12",
                              "e22", "e21", "e22", "a1", "lb1", "ub1", "lb2",
                              "ub2", "rms_error", "max_error"]
                lb = lower bound
                ub = upper bound
        """
        import pandas as pd

        bounds = []
        if d == 1:
            bounds.append(min(xdata))
            bounds.append(max(xdata))
        else:
            for i in range(d):
                bounds.append(min(xdata[i]))
                bounds.append(max(xdata[i]))

        if ftype != "ISMA":
            alphas = [alpha]

        data = hstack([[ftype, K, d], cs, exps, alphas, exp(bounds),
                       [rms_error, max_error]])
        df = pd.DataFrame(data).transpose()
        colnames = array(["ftype", "K", "d"])
        colnames = hstack([colnames, ["c%d" % k for k in range(1, K+1)]])
        colnames = hstack([colnames, ["e%d%d" % (k, i) for k in range(1, K+1)
                                      for i in range(1, d+1)]])
        colnames = hstack([colnames, ["a%d" % i for i in
                                      range(1, len(alphas)+1)]])
        colnames = hstack([colnames, hstack([["lb%d" % i, "ub%d" % i]
                                             for i in range(1, d+1)])])
        colnames = hstack([colnames, ["rms_err", "max_err"]])
        df.columns = colnames
        return df

    cstrt.get_dataframe = get_dataframe

    return cstrt, rms_error
