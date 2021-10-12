"""xfoil constraint set"""
import numpy as np
from gpfit.constraint_set import FitConstraintSet
from .wrapper import xfoil_comparison


# pylint: disable=too-many-arguments
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

        super().__init__(
            fitdata, ivar=ivar, dvars=dvars, name=name, err_margin=err_margin
        )

        self.airfoil = airfoil

    def process_result(self, result):
        """
        if data comes from Xfoil and airfoil is provided check against xfoil
        """
        super().process_result(result)

        if self.mfac not in result["sensitivities"]["constants"]:
            return
        if np.amax([abs(result["sensitivities"]["constants"][self.mfac])]) < 1e-5:
            return
        if not self.airfoil:
            return

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
            modelstr = ", ".join(self.ivar.descr["models"])
            msg = (
                f"Drag error for {modelstr} is {err[i]:.2f}. Re={re[i]:.1f};"
                f" CL={cl[i]:.4f}; Xfoil cd={cd[i]:.6f}, GP sol cd={cdx[i]:.6f}"
            )
            print(f"Warning: {msg}")
