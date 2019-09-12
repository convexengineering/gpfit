"Fit plotting"
from __future__ import division
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np
from gpfit.fit import fit
from gpfit.print_fit import print_MA, print_SMA

# pylint: disable=invalid-name
# pylint: disable=too-many-locals
def plot_fit_1d(udata, wdata, K=1, fitclass='MA', plotspace='log'):
    "Finds and plots a fit (MA or SMA) for 1D data"

    cstrt, _ = fit(np.log(udata), np.log(wdata), K, fitclass)
    uu = np.linspace(min(udata), max(udata), 1000)

    if fitclass == 'MA':
        uvarkey, = cstrt[0].left.varkeys
        A = [c.left.exps[0][uvarkey] for c in cstrt]
        B = np.log([c.left.cs[0] for c in cstrt])
        WW = []
        for k in range(K):
            WW += [np.exp(B[k])*uu**A[k]]
        stringlist = print_MA(A, B, 1, K)


    if fitclass == 'SMA':
        wexps, = cstrt.left.exps
        alpha, = list(wexps.values())
        uvarkey, = cstrt.right.varkeys
        A = [old_div(d[uvarkey],alpha) for d in cstrt.right.exps]
        B = old_div(np.log(cstrt.right.cs), alpha)

        ww = 0
        for k in range(K):
            ww += (np.exp(alpha*B[k])*uu**(alpha*A[k]))
        WW = [ww**(1./alpha)]

        print_str = print_SMA(A, B, alpha, 1, K)
        stringlist = [''.join(print_str)]

    f, ax = plt.subplots()
    if plotspace == 'log':
        ax.loglog(udata, wdata, '+r')
        for ww in WW:
            ax.loglog(uu, ww)
    elif plotspace == 'linear':
        ax.plot(udata, wdata, '+r')
        for ww in WW:
            ax.plot(uu, ww)
    ax.set_xlabel('u')
    ax.legend(['Data'] + stringlist, loc='best')

    plt.show()
    return f, ax

if __name__ == "__main__":
    N = 51
    U = np.logspace(0, np.log10(3), N)
    W = old_div((U**2+3), (U+1)**2)
    plot_fit_1d(U, W, K=2, fitclass='MA', plotspace="linear")
