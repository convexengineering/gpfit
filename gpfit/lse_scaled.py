from numpy import array, tile, exp, log

def lse_scaled(x, alpha):
    '''
    Log-sum-exponential function with derivatives
    - sums across the second dimension of x
    - note that lse_scaled is a mapping R^n --> R

    INPUTS:
            x:      independent variable data
                        2D numpy array [nPoints x nDim]

            alpha:  local softness parameter
                        1D array [K] (K=number of terms)

    OUTPUTS:
            y:      ISMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydx:   Deriv of y wrt each x
                        2D numpy array[nPoints x nDim]

            dydalpha:
                    [n-element 1D array], n is number of data points
    '''

    _, n = x.shape

    m = x.max(axis=1)  # maximal x values

    h = x - (tile(m, (n, 1))).T  # distance from m; note h <= 0 for all entries
    # (unless x has infs; in this case h has nans;
    # can prob deal w/ this gracefully)
    # should also deal with alpha==inf case gracefully

    expo = exp(alpha*h)
    sumexpo = expo.sum(axis=1)

    L = log(sumexpo)/alpha

    y = L + m

    dydx = expo/(tile(sumexpo, (n, 1))).T
    # note that sum(dydx,2)==1, i.e. dydx is a probability distribution
    dydalpha = ((h*expo).sum(axis=1)/sumexpo - L)/alpha

    return y, dydx, dydalpha
