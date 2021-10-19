"""Implements the fit function"""
from .classes import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine


# pylint: disable=too-many-arguments
def fit(xdata, ydata, K, fit_type="isma", alpha0=10, verbosity=0, seed=None):
    """A convenience function for returning a Fit object.

    Default behaviour returns the highest quality of fit (implicit softmax
    affine) but other options are max affine and softmax affine.

    Arguments
    ---------
    xdata: 2D numpy array [nDim, nPoints]
        Independent variable data

    ydata: 1D numpy array [nPoints,]
        Dependent variable data

    K: int
        Number of terms

    fit_type: str ("isma", "sma", "ma")
        Type of fit

    alpha0
        Initial guess for smoothing parameter alpha

    verbosity: int
        Verbosity

    seed: None or int
        Seed for random number generator in initialization function

    Returns
    -------
        Fit object

    """

    fits = {
        "ma": MaxAffine,
        "sma": SoftmaxAffine,
        "isma": ImplicitSoftmaxAffine,
    }
    return fits[fit_type](xdata, ydata, K, alpha0=alpha0, verbosity=verbosity,
                          seed=seed)
