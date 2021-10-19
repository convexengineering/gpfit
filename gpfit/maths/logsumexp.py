"Module for log-sum-exponential functions"
from numpy import zeros, spacing, exp, log, tile


# pylint: disable=too-many-locals
def lse_implicit(x, alpha):
    """
    Implicit Log-sum-exponential function with derivatives
    - sums across the second dimension of x
    - returns one y for every row of x
    - lse_implicit is a mapping R^n --> R, where n number of dimensions
    - implementation: newton raphson steps to find f(x,y) = 0

    Arguments:
    ----------
    x: 2D numpy array [nPoints x nDim]
        independent variable data

    alpha: 1D array [K] (K=number of terms)
        local softness parameter

    Returns:
    --------
    y: 1D numpy array [nPoints]
        ISMA approximation to log transformed data

    dydx: 2D numpy array[nPoints x K]
        Deriv of y wrt each x

    dydalpha: 2D array [nPoints x nDim]

    """

    tol = 10*spacing(1)
    npt, nx = x.shape

    if nx != alpha.size:
        raise ValueError("alpha size mismatch")

    alphamat = tile(alpha, (npt, 1))

    m = x.max(1)  # maximal x values
    # distance from m; note h <= 0 for all entries
    h = x - (tile(m, (nx, 1))).T
    L = zeros((npt,))  # initial guess. note y = m + L
    Lmat = (tile(L, (nx, 1))).T

    # initial eval
    expo = exp(alphamat*(h - Lmat))
    alphaexpo = alphamat*expo
    sumexpo = expo.sum(axis=1)
    sumalphaexpo = alphaexpo.sum(axis=1)
    f = log(sumexpo)
    dfdL = -alphaexpo.sum(axis=1)/sumexpo
    neval = 1
    i = abs(f) > tol  # inds to update

    while any(i):
        L[i] = L[i] - f[i]/dfdL[i]  # newton step
        # re-evaluate
        Lmat[i, :] = (tile(L[i], (nx, 1))).T
        expo[i, :] = exp(alphamat[i, :]*(h[i, :] - Lmat[i, :]))
        alphaexpo[i, :] = alphamat[i, :]*expo[i, :]
        sumexpo[i] = expo[i, :].sum(axis=1)
        sumalphaexpo[i, ] = alphaexpo[i, :].sum(axis=1)
        f[i] = log(sumexpo[i])
        dfdL[i] = -sumalphaexpo[i, ]/sumexpo[i]
        neval = neval + 1
        i[i] = abs(f[i]) > tol  # update inds that need to be evaluated

    y = m + L
    dydx = alphaexpo/(tile(sumalphaexpo, (nx, 1))).T
    dydalpha = (h - Lmat)*expo/(tile(sumalphaexpo, (nx, 1))).T

    return y, dydx, dydalpha


def lse_scaled(x, alpha):
    """
    Log-sum-exponential function with derivatives
    - sums across the second dimension of x
    - note that lse_scaled is a mapping R^n --> R

    Arguments:
    ----------
    x: 2D numpy array [nPoints x nDim]
        independent variable data

    alpha: 1D array [K] (K=number of terms)
        local softness parameter

    Returns:
    --------
    y: 1D numpy array [nPoints]
        ISMA approximation to log transformed data

    dydx: 2D numpy array[nPoints x nDim]
        Deriv of y wrt each x

    dydalpha: [n-element 1D array], n is number of data points

    """

    _, n = x.shape
    m = x.max(axis=1)  # maximal x values
    h = x - (tile(m, (n, 1))).T  # distance from m; note h <= 0 for all entries
    expo = exp(alpha*h)
    sumexpo = expo.sum(axis=1)
    L = log(sumexpo)/alpha
    y = L + m
    dydx = expo/(tile(sumexpo, (n, 1))).T
    # note that sum(dydx,2)==1, i.e. dydx is a probability distribution
    dydalpha = ((h*expo).sum(axis=1)/sumexpo - L)/alpha

    return y, dydx, dydalpha
