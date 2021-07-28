"""Implements the fit function"""
from .classes import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine


# pylint: disable=too-many-arguments
def fit(xdata, ydata, K, fit_type="isma", alpha0=10, verbosity=0):
    """A convenience function for returning a Fit object. Default behaviour
    returns the highest quality of fit (implicit softmax affine) but other
    options are max affine and softmax affine.

    Arguments
    ---------
        xdata:      Independent variable data
                        2D numpy array [nDim, nPoints]
                        [[<--------- x1 ------------->]
                         [<--------- x2 ------------->]]

        ydata:      Dependent variable data
                        1D numpy array [nPoints,]
                        [<---------- y ------------->]

        K:          Number of terms

        fit_type:   Type of fit ("isma", "sma", "ma")

        alpha0:     Initial guess for smoothing parameter alpha

        verbosity:  Verbosity

    Returns
    -------
        Fit object

    """

    fits = {
        "ma": MaxAffine,
        "sma": SoftmaxAffine,
        "isma": ImplicitSoftmaxAffine,
    }
    return fits[fit_type](xdata, ydata, K, alpha0=alpha0, verbosity=verbosity)
