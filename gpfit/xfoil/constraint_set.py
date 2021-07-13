from gpfit.constraint_set import FitConstraintSet


class XfoilFit(FitConstraintSet):
    """Special FitConstraintSet that can post-solve compare result to XFOIL

    Arguments (in addition to the arguments to FitConstraintSet)
    ---------
    airfoil:                            airfoil of fitted data
                                            str (e.g. "xxx.dat", "naca xxxx")

    """

    def __init__(
        self, fitdata, ivar=None, dvars=None, name="", err_margin=None, airfoil=False
    ):

        super(XfoilFit, self).__init__(
            fitdata, ivar=ivar, dvars=dvars, name=name, err_margin=err_margin
        )

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
            msg = (
                "Drag error for %s is %.2f. Re=%.1f; CL=%.4f;"
                " Xfoil cd=%.6f, GP sol cd=%.6f"
                % (
                    ", ".join(self.ivar.descr["models"]),
                    err[i],
                    re[i],
                    cl[i],
                    cd[i],
                    cdx[i],
                )
            )
            print("Warning: %s" % msg)
