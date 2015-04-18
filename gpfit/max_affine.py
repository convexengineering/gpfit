import numpy as np


def max_affine(x, ba):
    """
    Evaluates max affine function at values of x, given a set of
    max affine fit parameters.

    INPUTS
    ------
        x: 2D array [nPoints x nDim]
            Independent variable data

        ba: 2D array
            max affine fit parameters
            [[b1, a11, ... a1k]
             [ ....,          ]
             [bk, ak1, ... akk]]

    OUTPUTS
    -------
        y: 1D array [nPoints]
            Max affine output
        dydba: 2D array [nPoints x (nDim + 1)*K]
            dydba
    """
    npt, dimx = x.shape
    K = ba.size/(dimx + 1)
    ba = np.reshape(ba, (dimx + 1, K), order='F')  # 'F' gives Fortran indexing

    # augment data with column of ones
    X = np.hstack((np.ones((npt, 1)), x))

    y, partition = np.dot(X, ba).max(1), np.dot(X, ba).argmax(1)

    # The not-sparse sparse version
    dydba = np.zeros((npt, (dimx + 1)*K))
    for k in range(K):
        inds = np.equal(partition, k)
        indadd = (dimx + 1)*k
        ixgrid = np.ix_(inds.nonzero()[0], indadd + np.arange(dimx+1))
        dydba[ixgrid] = X[inds, :]

    return y, dydba
