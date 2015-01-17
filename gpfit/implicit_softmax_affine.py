from numpy import ones, reshape, nan, inf, hstack, dot, tile
from lse_implicit import lse_implicit
from repcols import repcols

def implicit_softmax_affine(x, params):
    '''
    Params come in as a vector (with alpha last)
    after reshaping (column-major), ba is dimx+1 by K
    first row is b
    rest is a

    INPUTS:
            x:      Independent variable data
                    [nx1 2D array]

            params: Fit parameters (Alpha is last element)
                    [2k+2? -element 1D array] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    OUTPUTS:
            y:
                    [n-element 1D array], n is number of data points

            dydp:
                    [n x 2k+2? 2D array]

    '''

    npt, dimx = x.shape

    K = params.size/(dimx+2)
    ba = params[0:-K]
    alpha = params[-K:]

    #reshape ba to matrix
    ba = ba.reshape(dimx+1, K, order='F') ######################################<<<<<<<

    if any(alpha <= 0):
        y = inf*ones((npt,1))
        dydp = nan 
        return y, dydp

    #augment data with column of ones
    X = hstack((ones((npt,1)), x))

    #compute affine functions
    z = dot(X,ba)


    y, dydz, dydalpha = lse_implicit(z, alpha)
    dydba = repcols(dydz, dimx+1) * tile(X, (1, K)) #<<<<<<<<<<<<<<<<<<
    dydp = hstack((dydba, dydalpha))

    return y, dydp