"Implements ba_init"
from numpy import ones, hstack, zeros, tile, argmin
from numpy.linalg import lstsq, matrix_rank
from numpy.random import permutation as randperm


# pylint: disable=too-many-locals
def ba_init(x, y, K):
    """
    Initializes max-affine fit to data (y, x)
    ensures that initialization has at least K+1 points per partition (i.e.
    per affine function)

    INPUTS:
        x:      Independent variable data
                    2D column vector [nPoints x nDims]

        y:      Dependent variable data
                    2D column vector [nPoints x 1]

    OUTPUTS:
        ba:     Initial b and a parameters
                    2D array [(dimx+1) x k]

    """
    defaults = {}
    defaults['bverbose'] = False
    options = defaults

    npt, dimx = x.shape

    X = hstack((ones((npt, 1)), x))
    b = zeros((dimx+1, K))

    if K*(dimx+1) > npt:
        raise ValueError('Not enough data points')

    # Choose K unique indices
    randinds = randperm(npt)[0:K]

    # partition based on distances
    sqdists = zeros((npt, K))
    for k in range(K):
        sqdists[:, k] = ((x - tile(x[randinds[k], :], (npt, 1))) ** 2).sum(1)

    # index to closest k for each data pt
    mindistind = argmin(sqdists, axis=1)

    # loop through each partition, making local fits
    # note we expand partitions that result in singular least squares problems
    # why this way? some points will be shared by multiple partitions, but
    # resulting max-affine fit will tend to be good. (as opposed to solving least-norm version)
    for k in range(K):
        inds = mindistind == k

        # before fitting, check rank and increase partition size if necessary
        # (this does create overlaps)
        if matrix_rank(X[inds, :]) < dimx + 1:
            sortdistind = sqdists[:, k].argsort()

            i = sum(inds)  # number of points in partition
            iinit = i

            if i < dimx+1:
                # obviously, at least need dimx+1 points. fill these in before
                # checking any ranks
                inds[sortdistind[i+1:dimx+1]] = 1  # TODO: check index
                i = dimx+1  # TODO: check index

            # now add points until rank condition satisfied
            while matrix_rank(X[inds, :]) < dimx+1:
                i = i+1
                inds[sortdistind[i]] = 1

            if options['bverbose']:
                print("ba_init: Added %s points to partition %s to maintain"
                      "full rank for local fitting." % (i-iinit, k))
        # now create the local fit
        b[:, k] = lstsq(X[inds.nonzero()], y[inds.nonzero()], rcond=-1)[0][:, 0]

    return b
