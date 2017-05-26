from gpkit import ConstraintSet
from gpkit import Variable, NomialArray, NamedVariables, VectorVariable
from gpkit.small_scripts import unitstr
# from xfoilWrapper import blind_call, single_cl
import numpy as np

class FitCS(ConstraintSet):
    def __init__(self, fitdata, ivar=None, dvars=None, nobounds=False,
                 err_margin=False, airfoil=False):

        self.airfoil = airfoil

        if ivar is None:
            with NamedVariables("fit"):
                dvars = VectorVariable(fitdata["d"], "u")
                ivar = Variable("w")

        self.dvars = dvars
        self.ivar = ivar

        monos = [fitdata["c%d" % k]*NomialArray(dvars**np.array(
            [fitdata["e%d%d" % (k, i)] for i in
             range(1, fitdata["d"]+1)])).prod(0) for k in
                 range(1, fitdata["K"]+1)]

        if err_margin == "Max":
            self.mfac = Variable("m_{fac-fit}", 1 + fitdata["max_err"], "-",
                                 "max error of fit")
        elif err_margin == "RMS":
            self.mfac = Variable("m_{fac-fit}", 1 + fitdata["rms_err"], "-",
                                 "RMS error of fit")
        else:
            self.mfac = Variable("m_{fac-fit}", 1.0, "-", "fit factor")

        if fitdata["ftype"] == "ISMA":
            # constraint of the form 1 >= c1*u1^exp1*u2^exp2*w^(-alpha) + ....
            alpha = np.array([fitdata["a%d" % k] for k in
                              range(1, fitdata["K"] + 1)])
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
            if not hasattr(rhs, "shape"):
                cstrt = (lhs == rhs)
        else:
            if hasattr(rhs, "shape"):
                cstrt = [(lh >= rh) for rh, lh in zip(rhs, lhs)]
            else:
                cstrt = (lhs >= rhs)

        self.constraint = cstrt
        if not hasattr(self.mfac, "__len__"):
            self.mfac = [self.mfac]*len(self.dvars)
        if not hasattr(self.ivar, "__len__"):
            self.ivar = [self.ivar]*len(self.dvars)

        self.bounds = []
        for dvar in self.dvars:
            bds = {}
            for i, v in enumerate(dvar):
                bds[v] = [float(fitdata["lb%d" % (i+1)].iloc[0]),
                          float(fitdata["ub%d" % (i+1)].iloc[0])]
            self.bounds.append(bds)

        ConstraintSet.__init__(self, self.constraint)

    def process_result(self, result, TOL=0.03):
        super(FitCS, self).process_result(result)

        if abs(result["sensitivities"]["constants"][self.mfac] < 1e-5):
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
            err, cdx = xfoil_comparison(self.airfoil, cl, re, cd)
            ind = np.where(err > TOL)[0]
            for i in ind:
                msg = ("Drag error for %s is %.2f. Re=%.1f; CL=%.4f;"
                       " Xfoil cd=%.6f, GP sol cd=%.6f" %
                       (", ".join(self.ivar.descr["models"]), err[i], re[i],
                        cl[i], cd[i], cdx[i]))
                print "Warning: %s" % msg
        else:
            for dvar in self.dvars:
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
                           (direct, num, bnd))
                    print "Warning: " + msg
