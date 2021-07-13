import numpy as np
from .logsumexp import lse_scaled, lse_implicit


def max_affine(x, ba):
    """
    Evaluates max affine function at values of x, given a set of
    max affine fit parameters.

    Arguments
    ---------
        x: 2D array [nPoints x nDim]
            Independent variable data

        ba: 2D array
            max affine fit parameters
            [[b1, a11, ... a1k]
             [ ....,          ]
             [bk, ak1, ... akk]]

    Returns
    -------
        y: 1D array [nPoints]
            Max affine output
        dydba: 2D array [nPoints x (nDim + 1)*K]
            dydba
    """
    npt, dimx = x.shape
    K = ba.size // (dimx + 1)
    ba = np.reshape(ba, (dimx + 1, K), order="F")  # 'F' gives Fortran indexing
    X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
    y, partition = np.dot(X, ba).max(1), np.dot(X, ba).argmax(1)

    dydba = np.zeros((npt, (dimx + 1)*K))
    for k in range(K):
        inds = np.equal(partition, k)
        indadd = (dimx + 1)*k
        ixgrid = np.ix_(inds.nonzero()[0], indadd + np.arange(dimx + 1))
        dydba[ixgrid] = X[inds, :]

    return y, dydba


# pylint: disable=too-many-locals
def softmax_affine(x, params):
    """
    Evaluates softmax affine function at values of x, given a set of
    SMA fit parameters.

    Arguments:
    ----------
            x:      Independent variable data
                        2D numpy array [nPoints x nDimensions]

            params: Fit parameters
                        1D numpy array [(nDim + 2)*K,]
                        [b1, a11, .. a1d, b2, a21, .. a2d, ...
                         bK, aK1, aK2, .. aKd, alpha]

    Returns:
    --------
            y:      SMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydp:   Jacobian matrix
    """

    npt, dimx = x.shape
    ba = params[0:-1]
    softness = params[-1]
    alpha = 1/softness
    if alpha <= 0:
        return np.inf*np.ones((npt, 1)), np.nan
    K = np.size(ba) // (dimx + 1)
    ba = ba.reshape(dimx + 1, K, order="F")
    X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
    z = np.dot(X, ba)  # compute affine functions
    y, dydz, dydsoftness = lse_scaled(z, alpha)

    dydsoftness = -dydsoftness*(alpha**2)
    nrow, ncol = dydz.shape
    repmat = np.tile(dydz, (dimx + 1, 1)).reshape(nrow, ncol*(dimx + 1), order="F")
    dydba = repmat*np.tile(X, (1, K))
    dydsoftness.shape = (dydsoftness.size, 1)
    dydp = np.hstack((dydba, dydsoftness))

    return y, dydp


# pylint: disable=too-many-locals
def implicit_softmax_affine(x, params):
    """
    Evaluates implicit softmax affine function at values of x, given a set of
    ISMA fit parameters.

    Arguments:
    ----------
            x:      Independent variable data
                        2D numpy array [nPoints x nDimensions]

            params: Fit parameters
                        1D numpy array [(nDim + 2)*K,]
                        [b1, a11, .. a1d, b2, a21, .. a2d, ...
                         bK, aK1, aK2, .. aKd, alpha1, alpha2, ... alphaK]

    Returns:
    --------
            y:      ISMA approximation to log transformed data
                        1D numpy array [nPoints]

            dydp:   Jacobian matrix

    """

    npt, dimx = x.shape
    K = params.size // (dimx + 2)
    ba = params[0:-K]
    alpha = params[-K:]
    if any(alpha <= 0):
        return np.inf*np.ones((npt, 1)), np.nan
    ba = ba.reshape(dimx + 1, K, order="F")  # reshape ba to matrix
    X = np.hstack((np.ones((npt, 1)), x))  # augment data with column of ones
    z = np.dot(X, ba)  # compute affine functions
    y, dydz, dydalpha = lse_implicit(z, alpha)

    nrow, ncol = dydz.shape
    repmat = np.tile(dydz, (dimx + 1, 1)).reshape(nrow, ncol*(dimx + 1), order="F")
    dydba = repmat*np.tile(X, (1, K))
    dydp = np.hstack((dydba, dydalpha))

    return y, dydp
