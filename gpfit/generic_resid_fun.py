def generic_resid_fun(yfun, xdata, ydata, params):
    '''
    generic residual function -- converts yfun(xdata,params)
    to a residfun [r, drdp] = generic_resid_fun(yfun, xdata, ydata, params)
    used by nonlinear least squares fitting algorithms
    to get a residual function [r,drdp] = residfun(params),
    use rfun = @(p) generic_resid_fun(@yfun, xdata, ydata, p)
        
    note this function defines resids as + when yhat > ydata 
    (opposite of typical conventions, but eliminates need for sign change)

    INPUTS:
        yfun:   Function (e.g. softmax_affine)

        xdata:  X data from original problem
                [n x 1 2D column vector] n = number of data points

        ydata:  Y data from original problem 
                [n x 1 2D column vector] n = number of data points

        params: Fit parameters
                [m-element 1D array] m = m(k) (e.g. For max-affine, m = 2*k)

    OUTPUTS:
        r:      residual [n-element 1D array] n = number of data points

        drdp:   Jacobian [n x m matrix]

    '''

    [yhat, drdp] = yfun(xdata, params)
    r = yhat - ydata.T[0] # Hacky way to perform elementwise subtraction between a row vector and a column vector

    return r, drdp