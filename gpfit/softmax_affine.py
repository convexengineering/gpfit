from numpy import size, inf, nan, ones, hstack, reshape, dot, tile
from lse_scaled import lse_scaled

def softmax_affine(x, params):
    '''
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
    '''

    ba = params[0:-1]
    softness = params[-1] #equivalent of end

    alpha = 1/softness

    npt, dimx = x.shape
    K = size(ba)/(dimx+1)
    ba = ba.reshape(dimx+1, K, order='F')

    if alpha <= 0:
        y = inf*ones((npt,1))
        dydp = nan
        return y, dydp

    #augment data with column of ones
    X = hstack((ones((npt,1)), x))

    #compute affine functions
    z = dot(X,ba)

    y, dydz, dydsoftness = lse_scaled(z, alpha)

    dydsoftness = - dydsoftness*(alpha**2)

    nrow, ncol = dydz.shape
    repmat = tile(dydz, (dimx+1, 1)).reshape(nrow, ncol*(dimx+1), order='F')
    dydba = repmat * tile(X, (1, K))

    dydsoftness.shape = (dydsoftness.size,1)
    dydp = hstack((dydba, dydsoftness))

    return y, dydp
