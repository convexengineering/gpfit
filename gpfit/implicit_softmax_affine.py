from numpy import ones, reshape, nan, inf, hstack, dot, tile
from lse_implicit import lse_implicit

def implicit_softmax_affine(x, params):
    '''
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

    '''

    npt, dimx = x.shape

    K = params.size/(dimx+2)
    ba = params[0:-K]
    alpha = params[-K:]

    #reshape ba to matrix
    ba = ba.reshape(dimx+1, K, order='F')

    if any(alpha <= 0):
        y = inf*ones((npt,1))
        dydp = nan 
        return y, dydp

    #augment data with column of ones
    X = hstack((ones((npt,1)), x))

    #compute affine functions
    z = dot(X,ba)


    y, dydz, dydalpha = lse_implicit(z, alpha)

    nrow, ncol = dydz.shape
    repmat = tile(dydz, (dimx+1, 1)).reshape(nrow, ncol*(dimx+1), order='F')
    dydba = repmat * tile(X, (1, K))
    dydp = hstack((dydba, dydalpha))

    return y, dydp