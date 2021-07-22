"""Fit constraint set"""
from numpy import amax, array, hstack
from gpkit import ConstraintSet
from gpkit import Variable, NomialArray, NamedVariables, VectorVariable
from gpkit.small_scripts import initsolwarning, appendsolwarning


# pylint: disable=too-many-instance-attributes, too-many-locals,
# pylint: disable=too-many-branches, no-member, import-error
# pylint: disable=too-many-arguments, import-outside-toplevel
class FitConstraintSet(ConstraintSet):
    """
    Constraint set for fitted functions

    Arguments
    ---------
    fitdata : dict
        dictionary of fit parameters
    ivar : gpkit Variable, Monomial, or NomialArray
        independent variable
    dvars : list of gpkit Variables, Monomials, or NomialArrays
        dependent variables
    err_margin : string, either "max" or "rms"
        flag to set margin factor using RMS or max error

    """

    def __init__(self, Fit, ivar=None, dvars=None, name="fit", err_margin=None):

        fitdata = Fit.fitdata

        if ivar is None:
            with NamedVariables("fit"):
                dvars = VectorVariable(fitdata["d"], "u")
                ivar = Variable("w")

        self.dvars = dvars
        self.ivar = ivar
        self.rms_err = Fit.error["rms"]
        self.max_err = Fit.error["max"]

        monos = [
            fitdata["c%d" % k]*NomialArray(array(dvars).T**array(
                [fitdata["e%d%d" % (k, i)] for i in range(fitdata["d"])]
                )).prod(NomialArray(dvars).ndim - 1) for k in range(fitdata["K"])
        ]

        with NamedVariables(name):
            self.mfac = Variable("mfac", 1.0, "-", "fit factor")
        if err_margin == "max":
            self.mfac.key.descr["value"] = 1 + self.max_err
        elif err_margin == "rms":
            self.mfac.key.descr["value"] = 1 + self.rms_err

        if type(Fit).__name__ == "ImplictSoftmaxAffine":
            # constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
            alpha = array([fitdata["a%d" % k] for k in range(fitdata["K"])])
            lhs, rhs = 1, NomialArray(monos/(ivar/self.mfac)**alpha).sum(0)
        elif type(Fit).__name__ == "SoftmaxAffine":
            # constraint of the form w^alpha >= c1*u1^exp1 + c2*u2^exp2 +....
            alpha = fitdata["a1"]
            lhs, rhs = (ivar/self.mfac)**alpha, NomialArray(monos).sum(0)
        elif type(Fit).__name__ == "MaxAffine":
            # constraint of the form w >= c1*u1^exp1, w >= c2*u2^exp2, ....
            lhs, rhs = (ivar/self.mfac), NomialArray(monos).T

        if fitdata["K"] == 1:
            # when possible, return an equality constraint
            if hasattr(rhs, "shape"):
                if rhs.ndim > 1:
                    self.constraint = [(lh == rh) for lh, rh in zip(lhs, rhs)]
                else:
                    self.constraint = lhs == rhs
            else:
                self.constraint = lhs == rhs
        else:
            if hasattr(rhs, "shape"):
                if rhs.ndim > 1:
                    self.constraint = [(lh >= rh) for lh, rh in zip(lhs, rhs)]
                else:
                    self.constraint = lhs >= rhs
            else:
                self.constraint = lhs >= rhs

        self.bounds = {}
        for i, dvar in enumerate(self.dvars):
            self.bounds[dvar] = [fitdata["lb%d" % i], fitdata["ub%d" % i]]

        ConstraintSet.__init__(self, [self.constraint])

    def process_result(self, result):
        """
        make sure fit result is within bounds of fitted data
        """
        super().process_result(result)
        initsolwarning(result, "Fit Out-of-Bounds")

        if self.mfac not in result["sensitivities"]["constants"]:
            return
        if amax([abs(result["sensitivities"]["constants"][self.mfac])]) < 1e-5:
            return

        for dvar in self.dvars:
            if isinstance(dvar, NomialArray):
                num = [result(x) for x in dvar]
            else:
                num = result(dvar)
            direct = None
            if any(x < self.bounds[dvar][0] for x in hstack([num])):
                direct, state = "lower", "below"
                bnd = self.bounds[dvar][0]
            if any(x > self.bounds[dvar][1] for x in hstack([num])):
                direct, state = "upper", "above"
                bnd = self.bounds[dvar][1]

            if direct:
                msg = (
                    "Variable %.100s could cause inaccurate result"
                    " because it is %s" % (dvar, state)
                    + " %s bound. Solution is %.4f but"
                    " bound is %.4f" % (direct, amax([num]), bnd)
                )
                appendsolwarning(msg, self, result, "Fit Out-of-Bounds")
