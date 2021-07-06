"Implements ISMA residual function"
from numpy import ones, nan, inf, hstack, dot, tile
from .lse_implicit import lse_implicit


# pylint: disable=too-many-locals
def implicit_softmax_affine(x, params):
    """
    Evaluates implicit softmax affine function at values of x, given a set of
    ISMA fit parameters.

    INPUTS:
            x:      Independent variable data
                        2D numpy array [nPoints x nDimensions]

            params: Fit parameters
                        1D numpy array [(nDim + 2)*K,]
                        [b1, a11, .. a1d, b2, a21, .. a2d, ...
                         bK, aK1, aK2, .. aKd, alpha1, alpha2, ... alphaK]

    OUTPUTS:
            y:      ISMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydp:   Jacobian matrix

    """

    npt, dimx = x.shape
    K = params.size//(dimx+2)
    ba = params[0:-K]
    alpha = params[-K:]
    if any(alpha <= 0):
        return inf*ones((npt, 1)), nan
    ba = ba.reshape(dimx+1, K, order='F')  # reshape ba to matrix

    X = hstack((ones((npt, 1)), x))  # augment data with column of ones
    z = dot(X, ba)  # compute affine functions

    y, dydz, dydalpha = lse_implicit(z, alpha)

    nrow, ncol = dydz.shape
    repmat = tile(dydz, (dimx+1, 1)).reshape(nrow, ncol*(dimx+1), order='F')
    dydba = repmat * tile(X, (1, K))
    dydp = hstack((dydba, dydalpha))

    return y, dydp
