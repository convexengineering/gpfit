" fit constraint set "
from __future__ import print_function
from __future__ import division
from numpy import amax, array, hstack, where
from gpkit import ConstraintSet
from gpkit import Variable, NomialArray, NamedVariables, VectorVariable
from numpy import amax, array, hstack, where

# pylint: disable=too-many-instance-attributes, too-many-locals,
# pylint: disable=too-many-branches, no-member, import-error
# pylint: disable=too-many-arguments


class FitCS(ConstraintSet):
    """Constraint set for fitted functions

    Arguments
    ---------
    fitdata : dict
        dictionary of fit parameters
    ivar : gpkit Variable, Monomial, or NomialArray
        independent variable
    dvars : list of gpkit Variables, Monomials, or NomialArrays
        dependent variables
    err_margin : string, either "Max" or "RMS"
        flag to set margin factor using RMS or max error

    """
    def __init__(self, fitdata, ivar=None, dvars=None, name="",
                 err_margin=None):

        self.fitdata = fitdata

        if ivar is None:
            with NamedVariables("fit"):
                dvars = VectorVariable(fitdata["d"], "u")
                ivar = Variable("w")

        self.dvars = dvars
        self.ivar = ivar
        self.rms_err = fitdata["rms_err"]
        self.max_err = fitdata["max_err"]

        monos = [fitdata["c%d" % k]*NomialArray(array(dvars).T**array(
            [fitdata["e%d%d" % (k, i)] for i in
             range(fitdata["d"])])).prod(NomialArray(dvars).ndim - 1)
                 for k in range(fitdata["K"])]

        if err_margin == "Max":
            self.mfac = Variable("m_{fac-" + name + "-fit}",
                                 1 + self.max_err, "-", "max error of fit")
        elif err_margin == "RMS":
            self.mfac = Variable("m_{fac-" + name + "-fit}",
                                 1 + self.rms_err, "-", "RMS error of fit")
        elif err_margin is None:
            self.mfac = Variable("m_{fac-" + name + "-fit}", 1.0, "-",
                                 "fit factor")
        else:
            raise ValueError("Invalid name for err_margin: valid inputs Max, "
                             "RMS")

        if fitdata["ftype"] == "ISMA":
            # constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
            alpha = array([fitdata["a%d" % k] for k in
                           range(fitdata["K"])])
            lhs, rhs = 1, NomialArray(monos/(ivar/self.mfac)**alpha).sum(0)
        elif fitdata["ftype"] == "SMA":
            # constraint of the form w^alpha >= c1*u1^exp1 + c2*u2^exp2 +....
            alpha = fitdata["a1"]
            lhs, rhs = (ivar/self.mfac)**alpha, NomialArray(monos).sum(0)
        elif fitdata["ftype"] == "MA":
            # constraint of the form w >= c1*u1^exp1, w >= c2*u2^exp2, ....
            lhs, rhs = (ivar/self.mfac), NomialArray(monos).T

        if fitdata["K"] == 1:
            # when possible, return an equality constraint
            if hasattr(rhs, "shape"):
                if rhs.ndim > 1:
                    self.constraint = [(lh == rh) for lh, rh in zip(lhs, rhs)]
                else:
                    self.constraint = (lhs == rhs)
            else:
                self.constraint = (lhs == rhs)
        else:
            if hasattr(rhs, "shape"):
                if rhs.ndim > 1:
                    self.constraint = [(lh >= rh) for lh, rh in zip(lhs, rhs)]
                else:
                    self.constraint = (lhs >= rhs)
            else:
                self.constraint = (lhs >= rhs)

        self.bounds = {}
        for i, dvar in enumerate(self.dvars):
            self.bounds[dvar] = [fitdata["lb%d" % i],
                                 fitdata["ub%d" % i]]

        ConstraintSet.__init__(self, [self.constraint])

    def get_fitdata(self):
        " return fit data "
        return self.fitdata

    def get_dataframe(self):
        " return a pandas DataFrame of fit parameters "
        import pandas as pd
        df = pd.DataFrame(list(self.fitdata.values())).transpose()
        df.columns = list(self.fitdata.keys())
        return df

    def process_result(self, result):
        """
        make sure fit result is within bounds of fitted data
        """
        super(FitCS, self).process_result(result)

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
                msg = ("Variable %.100s could cause inaccurate result"
                       " because it is %s" % (dvar, state)
                       + " %s bound. Solution is %.4f but"
                       " bound is %.4f" %
                       (direct, amax([num]), bnd))
                print("Warning: " + msg)


class XfoilFit(FitCS):
    """Special FitCS that can post-solve compare result to XFOIL

    Arguments (in addition to the arguments to FitCS)
    ---------
    airfoil:                            airfoil of fitted data
                                            str (e.g. "xxx.dat", "naca xxxx")

    """
    def __init__(self, fitdata, ivar=None, dvars=None, name="",
                 err_margin=None, airfoil=False):

        super(XfoilFit, self).__init__(fitdata, ivar=ivar, dvars=dvars,
                                       name=name, err_margin=err_margin)

        self.airfoil = airfoil

    def process_result(self, result):
        """
        if data comes from Xfoil and airfoil is provided check against xfoil
        """
        super(XfoilFit, self).process_result(result)

        if self.mfac not in result["sensitivities"]["constants"]:
            return
        if amax([abs(result["sensitivities"]["constants"][self.mfac])]) < 1e-5:
            return
        if not self.airfoil:
            return

        from .xfoilWrapper import xfoil_comparison
        cl, re = 0.0, 0.0
        for dvar in self.dvars:
            if "Re" in str(dvar):
                re = result(dvar)
            if "C_L" in str(dvar):
                cl = result(dvar)
        cd = result(self.ivar)
        if not hasattr(cl, "__len__") and hasattr(re, "__len__"):
            cl = [cl]*len(re)
        err, cdx = xfoil_comparison(self.airfoil, cl, re, cd)
        ind = where(err > 0.05)[0]
        for i in ind:
            msg = ("Drag error for %s is %.2f. Re=%.1f; CL=%.4f;"
                   " Xfoil cd=%.6f, GP sol cd=%.6f" %
                   (", ".join(self.ivar.descr["models"]), err[i], re[i],
                    cl[i], cd[i], cdx[i]))
            print("Warning: %s" % msg)
