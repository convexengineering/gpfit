" python wrapper to call xfoil "
import subprocess
import numpy as np


# pylint: disable=invalid-name, bare-except, too-many-locals
# pylint: disable=too-many-arguments, consider-using-with
def xfoil_comparison(airfoil, Cl, Re, Cd):
    """
    Comparison of XFOIL Cd to input Cd for given Cl and Re.

    Arguments
    ---------
    airfoil:                    airfoil name
                                    str - ("xxx.dat", "naca xxxx")
    Cl:                         lift coefficient
                                    float or 1d-list or array
    Re:                         Reynolds number
                                    float or 1d-list or array
    Cd:                         drag coefficient
                                    float or 1d-list or array

    Returns
    -------
    err:                        error between Cd and XFOIL computed Cd
                                    1-d numpy array
    cdxs:                       XFOIL computed Cd values
                                    1-d numpy array
    """

    if ".dat" in airfoil:
        topline = "load " + airfoil + " \n afl \n"
    elif "naca" in airfoil:
        topline = airfoil + "\n"
    else:
        raise ValueError("Invalid airfoil. Valid types are *.dat, naca xxxx")

    if not hasattr(Cl, "__len__"):
        Cl, Re, Cd = [Cl], [Re], [Cd]

    err, cdxs = [], []
    for cl, re, cd in zip(Cl, Re, Cd):
        failmsg = f"Xfoil call failed at CL={cl:.4f} and Re={re:.1f}"
        try:
            x = single_call(topline, cl, re, 0.0)
            if "VISCAL:  Convergence failed" in x:
                print(f"Convergence Warning: {failmsg}")
                cdx = cd
            else:
                cdx = x[0]
        except subprocess.CalledProcessError:
            print(f"Unable to start Xfoil: {failmsg}")
            cdx = cd

        err.append(1 - cd/cdx)
        cdxs.append(cdx)

    return np.array(err), np.array(cdxs)


def single_call(topline, cl, Re, M, max_iter=100, pathname="/usr/local/bin/xfoil"):
    """
    Single XFOIL call for given Cl, Re and Mach number.

    Arguments
    ---------
    topline:                        load airfoil call in XFOIL
                                        str - e.g. "load xxx.dat"
    cl:                             lift coefficient
                                        float
    Re:                             Reynolds number
                                        float
    M:                              Mach number
                                        float
    max_iter:                       number of XFOIL iterations
                                        int: default is 100
    pathname:                       system path to XFOIL
                                        str
    Returns
    -------
    cd:                         XFOIL drag coefficient
                                    float
    cl:                         XFOIL lift coefficient
                                    float
    cm:                         XFOIL moment coefficient
                                    float
    std_out:                    XFOIL output
                                    str
    """

    proc = subprocess.Popen([pathname], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    proc.stdin.write(
        topline
        + "oper \n"
        + f"iter {max_iter}\n"
        + "visc \n"
        + f"{Re:.2e} \n"
        + "M \n"
        + f"{M:.2f} \n"
        + "a 2.0 \n"
        + f"cl {cl:.4f} \n"
        + "\n"
        + "quit \n"
    )
    stdout_val = proc.communicate()[0]
    proc.stdin.close()

    if "VISCAL:  Convergence failed\n" in stdout_val:
        return stdout_val

    res = {}
    ostr = stdout_val.split()
    ctr = 0
    for i in range(0, len(ostr)):
        ix = len(ostr) - (i + 1)
        vl = ostr[ix]
        if vl in ["a", "CL", "CD", "Cm"]:
            res[vl] = ostr[ix + 2]
            ctr += 1
        if ctr >= 4:
            break
    cd = res["CD"]
    cl = res["CL"]
    # alpha_ret = res['a']
    cm = res["Cm"]
    return float(cd), float(cl), float(cm), stdout_val
