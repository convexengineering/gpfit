from numpy import append, ones, exp, sqrt, mean, square
from gpkit.nomials import (Posynomial, Monomial, PosynomialInequality,
                           MonomialEquality)
from implicit_softmax_affine import implicit_softmax_affine
from softmax_affine import softmax_affine
from max_affine import max_affine
from LM import LM
from generic_resid_fun import generic_resid_fun
from max_affine_init import max_affine_init
from print_fit import print_ISMA, print_SMA, print_MA

def fit(xdata, ydata, K, ftype="ISMA", varNames=None):
    '''
    Fits a log-convex function to multivariate data and returns a GP-compatible constraint

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

        varNames:   List of variable names strings
                        Independent variables first, with dependent variable at the end
                        Default value: ['u_1', 'u_2', ...., 'u_d', 'w']

    OUTPUTS
        cstrt:      GPkit PosynomialInequality object
                        For K > 1, this will be a posynomial inequality constraint
                        If K = 1, this is automatically made into an equality constraint
                        If MA fit and K > 1, this is a list of constraint objects

        rmsErr:     RMS error
                        Root mean square error between original (not log transformed)
                        data and function fit.
    '''

    # Check data is in correct form
    if ydata.ndim > 1:
        raise ValueError('Dependent data should be in the form of a 1D numpy array')

    # Transform to column vector
    ydata = ydata.reshape(ydata.size,1)

    if xdata.ndim == 1:
        xdata = xdata.reshape(xdata.size,1)
    else:
        xdata = xdata.T

    # Dimension of function (number of independent variables)
    d = xdata.shape[1]

    # Create varNames if None
    if varNames == None:
        varNames = []
        for i in range(d):
            varNames.append('u_{0:d}'.format(i+1))
        varNames.append('w')

    # Initialize fitting variables
    alphainit = 10
    bainit = max_affine_init(xdata, ydata, K)

    if ftype == "ISMA":

        def rfun (p):
            return generic_resid_fun(implicit_softmax_affine, xdata, ydata, p)
        [params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit*ones((K,1))))

        # Approximated data
        y_ISMA, _ = implicit_softmax_affine(xdata, params)
        w_ISMA = exp(y_ISMA)

        # RMS error
        w = (exp(ydata)).T[0]
        #rmsErr = sqrt(mean(square(w_ISMA-w)))
        rmsErr = sqrt(mean(square(y_ISMA-ydata.T[0])))

        alpha = 1./params[range(-K,0)]

        # A: exponent parameters, B: coefficient parameters
        A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

        print_str = print_ISMA(A, B, alpha, d, K)

        # Calculate c's and exp's for creating GPkit objects
        cs = []
        exps = []
        for k in range(K):
            cs.append(exp(alpha[k] * B[k]))
            expdict = {}
            for i in range(d):
                expdict[varNames[i]] = alpha[k] * A[k*d + i]
            expdict[varNames[-1]] = -alpha[k]
            exps.append(expdict)

        cs = tuple(cs)
        exps = tuple(exps)

        # Create gpkit objects
        # ISMA returns a constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
        posy = Posynomial(exps, cs)
        cstrt = (posy <= 1)

        # # If only one term, automatically make an equality constraint
        if K == 1:
            cstrt = MonomialEquality(cstrt, "=", 1)

    elif ftype == "SMA":

        def rfun (p):
            return generic_resid_fun(softmax_affine, xdata, ydata, p)
        [params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit))

        # Approximated data
        y_SMA, _ = softmax_affine(xdata, params)
        w_SMA = exp(y_SMA)

        # RMS error
        w = (exp(ydata)).T[0]
        #rmsErr = sqrt(mean(square(w_SMA-w)))
        rmsErr = sqrt(mean(square(y_SMA-ydata.T[0])))

        alpha = 1./params[-1]

        A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

        print_str = print_SMA(A, B, alpha, d, K)

        # Calculate c's and exp's for creating GPkit objects
        cs = []
        exps = []
        for k in range(K):
            cs.append(exp(alpha * B[k]))
            expdict = {}
            for i in range(d):
                expdict[varNames[i]] = alpha * A[k*d + i]
            exps.append(expdict)

        cs = tuple(cs)
        exps = tuple(exps)

        # Creates dictionary for the monomial side of the constraint
        w_exp = {varNames[-1]: alpha}

        # Create gpkit objects
        # SMA returns a constraint of the form w^alpha >= c1*u1^exp1 + c2*u2^exp2 +....
        posy  = Posynomial(exps, cs)
        mono = Monomial(w_exp, 1)
        cstrt = (mono >= posy)

        # # If only one term, automatically make an equality constraint
        if K == 1:
            cstrt = MonomialEquality(cstrt, "=", 1)


    elif ftype == "MA":

        def rfun(p):
            return generic_resid_fun(max_affine, xdata, ydata, p)
        [params, RMStraj] = LM(rfun, bainit.flatten('F'))

        # Approximated data
        y_MA, _ = max_affine(xdata, params)
        w_MA = exp(y_MA)

        # RMS error
        w = (exp(ydata)).T[0]
        #rmsErr = sqrt(mean(square(w_MA-w)))
        rmsErr = sqrt(mean(square(y_MA-ydata.T[0])))

        A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

        print_str = print_MA(A, B, d, K)

        w_exp = {varNames[-1]: 1}
        mono1 = Monomial(w_exp,1)

        cstrt = []
        for k in range(K):
            cs = exp(B[k])

            exps = {}
            for i in range(d):
                exps[varNames[i]] = A[k*d + i]
            mono2 = Monomial(exps, cs)
            cstrt1 = PosynomialInequality(mono2, "<=", mono1)
            cstrt.append(cstrt1)

        if K == 1:
            cstrt = MonomialEquality(mono2, "=", mono1)

    return cstrt, rmsErr

