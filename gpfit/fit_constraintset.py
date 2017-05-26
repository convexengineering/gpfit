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

        ConstraintSet.__init__(self, self.constraints)

    def process_result(self, result, TOL=0.03):
        super(FitCS, self).process_result(result)

        for mfac, dvrs, ivr, bds in zip(self.mfac, self.dvars, self.ivar,
                                        self.bounds):

            if self.airfoil:
                runxfoil = True
                if ".dat" in self.airfoil:
                    topline = "load " + self.airfoil + " \n afl \n"
                elif "naca" in self.airfoil:
                    topline = self.airfoil + "\n"
                else:
                    print "Bad airfoil specified"
            else:
                runxfoil = False
            bndwrn = True

            if abs(result["sensitivities"]["constants"][mfac]) < 1e-5:
                bndwrn = False
                runxfoil = False

            if runxfoil:
                cl, re = 0.0, 0.0
                for d in dvrs:
                    if "C_L" in str(d):
                        cl = result(d)
                    if "Re" in str(d):
                        re = result(d)
                cdgp = result(ivr)
                failmsg = "Xfoil call failed at CL=%.4f and Re=%.1f" % (cl, re)
                try:
                    x = blind_call(topline, cl, re, 0.0)
                    if "VISCAL:  Convergence failed" in x:
                        print "Convergence Warning: %s" % failmsg
                        cd, cl = cdgp, 1.0
                    else:
                        cd, cl = x[0], x[1]
                except:
                    print "Unable to start Xfoil: %s" % failmsg
                    cd, cl = cdgp, 1.0

                err = 1 - cdgp/cd
                if err > TOL:
                    msg = ("Drag error for %s is %.2f. Re=%.1f; CL=%.4f;"
                           " Xfoil cd=%.6f, GP sol cd=%.6f" %
                           (", ".join(d.descr["models"]), err, re, cl, cd,
                            cdgp))
                    print "Warning: %s" % msg
                else:
                    bndwrn = False

            if bndwrn:
                for d in dvrs:
                    num = result(d)
                    err = 0.0
                    if num < bds[d][0]:
                        direct = "lower"
                        bnd = bds[d][0]
                        err = num/bnd
                    if num > bds[d][1]:
                        direct = "upper"
                        bnd = bds[d][1]
                        err = 1 - num/bnd

                    if err > TOL:
                        msg = ("Variable %.100s could cause inaccurate result"
                               " because it exceeds" % d
                               + " %s bound. Solution is %.4f but"
                               " bound is %.4f" %
                               (direct, num, bnd))
                        print "Warning: " + msg
