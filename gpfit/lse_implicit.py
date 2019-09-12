"Implements lse_implicit"
from __future__ import print_function
from __future__ import division
from past.utils import old_div
from numpy import zeros, spacing, exp, log, tile


# pylint: disable=too-many-locals
def lse_implicit(x, alpha):
    """
    Implicit Log-sum-exponential function with derivatives
    - sums across the second dimension of x
    - returns one y for every row of x
    - lse_implicit is a mapping R^n --> R, where n number of dimensions
    - implementation: newton raphson steps to find f(x,y) = 0

    INPUTS:
            x:      independent variable data
                        2D numpy array [nPoints x nDim]

            alpha:  local softness parameter
                        1D array [K] (K=number of terms)

    OUTPUTS:
            y:      ISMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydx:   Deriv of y wrt each x
                        2D numpy array[nPoints x K]

            dydalpha:
                        2D array [nPoints x nDim]
    """

    bverbose = False

    tol = 10*spacing(1)
    npt, nx = x.shape

    if nx != alpha.size:
        raise ValueError('alpha size mismatch')

    alphamat = tile(alpha, (npt, 1))

    m = x.max(1)  # maximal x values
    # distance from m; note h <= 0 for all entries
    h = x - (tile(m, (nx, 1))).T
    # (unless x has infs; in this case h has nans;
    # can prob deal w/ this gracefully)
    # should also deal with alpha==inf case gracefully

    L = zeros((npt,))  # initial guess. note y = m + L

    Lmat = (tile(L, (nx, 1))).T

    # initial eval
    expo = exp(alphamat*(h-Lmat))
    alphaexpo = alphamat*expo
    sumexpo = expo.sum(axis=1)
    sumalphaexpo = alphaexpo.sum(axis=1)
    f = log(sumexpo)
    dfdL = old_div(-alphaexpo.sum(axis=1),sumexpo)
    neval = 1
    i = abs(f) > tol  # inds to update

    while any(i):
        L[i] = L[i] - old_div(f[i],dfdL[i])    # newton step
        # re-evaluate
        Lmat[i, :] = (tile(L[i], (nx, 1))).T
        expo[i, :] = exp(alphamat[i, :] * (h[i, :] - Lmat[i, :]))
        alphaexpo[i, :] = alphamat[i, :] * expo[i, :]
        sumexpo[i] = expo[i, :].sum(axis=1)
        sumalphaexpo[i, ] = alphaexpo[i, :].sum(axis=1)
        f[i] = log(sumexpo[i])
        dfdL[i] = old_div(-sumalphaexpo[i, ],sumexpo[i])
        neval = neval + 1

        # update inds that need to be evaluated
        i[i] = abs(f[i]) > tol

    if bverbose:
        print('lse_implicit converged in ' +
              repr(neval) + ' newton-raphson steps')

    y = m + L

    dydx = old_div(alphaexpo,(tile(sumalphaexpo, (nx, 1))).T)
    dydalpha = old_div((h - Lmat)*expo,(tile(sumalphaexpo, (nx, 1))).T)

    return y, dydx, dydalpha
