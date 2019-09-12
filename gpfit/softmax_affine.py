"Implements SMA residual function"
from __future__ import division
from past.utils import old_div
from numpy import size, inf, nan, ones, hstack, dot, tile
from .lse_scaled import lse_scaled


# pylint: disable=too-many-locals
def softmax_affine(x, params):
    """
    Evaluates softmax affine function at values of x, given a set of
    SMA fit parameters.

    INPUTS:
            x:      Independent variable data
                        2D numpy array [nPoints x nDimensions]

            params: Fit parameters
                        1D numpy array [(nDim + 2)*K,]
                        [b1, a11, .. a1d, b2, a21, .. a2d, ...
                         bK, aK1, aK2, .. aKd, alpha]

    OUTPUTS:
            y:      SMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydp:   Jacobian matrix
    """

    npt, dimx = x.shape
    ba = params[0:-1]
    softness = params[-1]
    alpha = old_div(1,softness)
    if alpha <= 0:
        return inf*ones((npt, 1)), nan
    K = old_div(size(ba),(dimx+1))
    ba = ba.reshape(dimx+1, K, order='F')

    X = hstack((ones((npt, 1)), x))  # augment data with column of ones
    z = dot(X, ba)  # compute affine functions

    y, dydz, dydsoftness = lse_scaled(z, alpha)

    dydsoftness = -dydsoftness*(alpha**2)

    nrow, ncol = dydz.shape
    repmat = tile(dydz, (dimx+1, 1)).reshape(nrow, ncol*(dimx+1), order='F')
    dydba = repmat * tile(X, (1, K))
    dydsoftness.shape = (dydsoftness.size, 1)
    dydp = hstack((dydba, dydsoftness))

    return y, dydp
