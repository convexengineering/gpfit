"Fit plotting"
import numpy as np
import matplotlib.pyplot as plt
from gpfit.fit import fit
from gpfit.print_fit import print_MA, print_SMA

def plot_fit_1d(u, w, K=1, fitclass='MA', plotspace='log'):
    "Finds and plots a fit (MA or SMA) for 1D data"

    x = np.log(u)
    y = np.log(w)

    cstrt, _ = fit(x, y, K, fitclass)

    uu = np.linspace(min(u), max(u), 1000)
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
        alpha, = wexps.values()
        uvarkey, = cstrt.right.varkeys
        A = [d[uvarkey]/alpha for d in cstrt.right.exps]
        B = np.log(cstrt.right.cs) / alpha

        ww = 0
        for k in range(K):
            ww += (np.exp(alpha*B[k])*uu**(alpha*A[k]))
        WW = [ww**(1./alpha)]

        print_str = print_SMA(A, B, alpha, 1, K)
        stringlist = [''.join(print_str)]

    f, ax = plt.subplots()
    if plotspace == 'log':
        ax.loglog(u, w, '+r')
        for ww in WW:
            ax.loglog(uu, ww)
    elif plotspace == 'linear':
        ax.plot(u, w, '+r')
        for ww in WW:
            ax.plot(uu, ww)
    ax.set_xlabel('u')
    ax.legend(['Data'] + stringlist,
               loc='best')

    plt.show()
    return f, ax

if __name__ == "__main__":
    n = 51
    u = np.logspace(0, np.log10(3), n)
    w = (u**2+3) / (u+1)**2
    f = plot_fit_1d(u, w, K=2, fitclass='MA', plotspace="linear")
