from numpy import zeros, hstack, reshape, ones, dot, arange, equal, ix_


def max_affine(x, ba, flag=False):
    '''
    Evaluates max affine function at values of x, given a set of
    max affine fit parameters. 

    INPUTS:
            x:      Independent variable data
                        2D array [n x 1]

            ba:     max affine fit parameters
                        2D array [dimx+1]

    OUTPUTS:
            y:      MA approximation to log transformed data
                        1D array [nPoints]

            dydba:  dydba
                        2D array [nPoints x (nDim + 1)*K]

    '''
    npt, dimx = x.shape
    K = ba.size/(dimx+1)
    ba = reshape(ba, (dimx+1, K), order='F')  # 'F' gives Fortran indexing

    # augment data with column of ones
    X = hstack((ones((npt, 1)), x))

    y, partition = (dot(X, ba)).max(1), (dot(X, ba)).argmax(1)

    # The not-sparse sparse version
    dydba = zeros((npt, (dimx + 1)*K))
    for k in range(K):
        inds = equal(partition, k)
        indadd = (dimx+1)*k
        ixgrid = ix_(inds.nonzero()[0], indadd + arange(dimx+1))
        dydba[ixgrid] = X[inds, :]

    return y, dydba
