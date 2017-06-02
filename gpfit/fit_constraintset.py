from gpkit import ConstraintSet
from gpkit import Variable, NomialArray, NamedVariables, VectorVariable
from gpkit.small_scripts import unitstr
import numpy as np

class FitCS(ConstraintSet):
    def __init__(self, fitdata, ivar=None, dvars=None, err_margin=None,
                 airfoil=False):

        self.airfoil = airfoil
        self.fitdata = fitdata

        if ivar is None:
            with NamedVariables("fit"):
                dvars = VectorVariable(fitdata["d"], "u")
                ivar = Variable("w")

        self.dvars = dvars
        self.ivar = ivar
        self.rms_err = fitdata["rms_err"]
        self.max_err = fitdata["max_err"]

        monos = [fitdata["c%d" % k]*NomialArray(np.array(dvars).T**np.array(
            [fitdata["e%d%d" % (k, i)] for i in
             range(fitdata["d"])])).prod(NomialArray(dvars).ndim - 1) for k in
                 range(fitdata["K"])]

        if err_margin == "Max":
            self.mfac = Variable("m_{fac-fit}", 1 + self.max_err, "-",
                                 "max error of fit")
        elif err_margin == "RMS":
            self.mfac = Variable("m_{fac-fit}", 1 + self.rms_err, "-",
                                 "RMS error of fit")
        elif err_margin != None:
            raise ValueError("Invalid name for err_margin: valid inputs Max, "
                             "RMS")
        else:
            self.mfac = Variable("m_{fac-fit}", 1.0, "-", "fit factor")

        if fitdata["ftype"] == "ISMA":
            # constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
            alpha = np.array([fitdata["a%d" % k] for k in
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
        df = pd.DataFrame(self.fitdata.values()).transpose()
        df.columns = self.fitdata.keys()
        return df

    def process_result(self, result):
        """
        make sure fit result is within bounds of fitted data
        if data comes from Xfoil and airfoil is provided check against xfoil
        """
        super(FitCS, self).process_result(result)

        if np.amax([abs(result["sensitivities"]["constants"][self.mfac])]) < 1e-5:
            return None

        if self.airfoil:
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
            ind = np.where(err > 0.05)[0]
            for i in ind:
                msg = ("Drag error for %s is %.2f. Re=%.1f; CL=%.4f;"
                       " Xfoil cd=%.6f, GP sol cd=%.6f" %
                       (", ".join(self.ivar.descr["models"]), err[i], re[i],
                        cl[i], cd[i], cdx[i]))
                print "Warning: %s" % msg
        else:
            for dvar in self.dvars:
                if isinstance(dvar, NomialArray):
                    num = [result(x) for x in dvar]
                else:
                    num = result(dvar)
                direct = None
                if any(x < self.bounds[dvar][0] for x in np.hstack([num])):
                    direct = "lower"
                    bnd = self.bounds[dvar][0]
                if any(x > self.bounds[dvar][1] for x in np.hstack([num])):
                    direct = "upper"
                    bnd = self.bounds[dvar][1]

                if direct:
                    msg = ("Variable %.100s could cause inaccurate result"
                           " because it exceeds" % dvar
                           + " %s bound. Solution is %.4f but"
                           " bound is %.4f" %
                           (direct, np.amax([num]), bnd))
                    print "Warning: " + msg
