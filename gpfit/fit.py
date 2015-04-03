from implicit_softmax_affine import implicit_softmax_affine
from LM import LM
from generic_resid_fun import generic_resid_fun
from max_affine_init import max_affine_init
from print_fit import print_ISMA, print_SMA, print_MA
from gpkit.nomials import Posynomial, Constraint, MonoEQConstraint
from numpy import append, ones, exp, sqrt, mean, square

def fit(xdata, ydata, K, ftype="ISMA", varNames=None):
    '''
    Fits a log-convex function to multivariate data and returns a GP-compatible constraint

    INPUTS
        xdata:        Independent variable data 
                    2D numpy array [nDim, nPoints]

        ydata:        Dependent variable data
                    1D numpy array [nPoints,]

        K:            Number of terms in the fit

        ftype:        Fit type
                    "ISMA" (default) or "SMA" or "MA"

        varNames:    Variable names (list)
                    independent variables first, with dependent variable at the end
                    Default: [u1, u2, ...., uN, w]

    OUTPUTS
        cstrt:        GPkit constraint object
                    If K = 1, this is automatically made into an equality constraint

        rmsErr:        RMS Error
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

        def rfun (p): return generic_resid_fun(implicit_softmax_affine, xdata, ydata, p)
        [params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit*ones((K,1))))

        # Approximated data
        y_ISMA, _ = implicit_softmax_affine(xdata, params)
        w_ISMA = exp(y_ISMA)

        # RMS error
        w = (exp(ydata)).T[0]
        rmsErr = sqrt(mean(square(w_ISMA-w)))

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
        posy  = Posynomial(exps, cs)
        cstrt = Constraint(posy,1)

        # # If only one term, automatically make an equality constraint
        if K == 1:
            cstrt = MonoEQConstraint(cstrt,1)

    elif ftype == "SMA":

        def rfun (p): return generic_resid_fun(softmax_affine, xdata, ydata, p)
        [params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit))

        alpha = 1./params[-1]

        A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

        print_str = print_SMA(A, B, alpha, d, K)


    elif ftype == "MA":

        A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
        B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

        print_str = print_MA(A, B, d, K)


    return cstrt, rmsErr

