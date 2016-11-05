from gpfit.fit import fit
from gpfit.print_fit import print_MA, print_SMA
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_1d(u, w, K=1, fitclass='MA', plotspace='log'):
    "Finds and plots a fit (MA or SMA) for 1D data"

    x = np.log(u)
    y = np.log(w)

    params, _ = fit(x, y, K, fitclass)

    uu = np.linspace(min(u), max(u), 100)
    if fitclass == 'MA':
        A = params[[i for i in range(K*2) if i % 2 != 0]]
        B = params[[i for i in range(K*2) if i % 2 == 0]]
        WW = []
        for k in range(K):
            WW += [np.exp(B[k])*uu**A[k]]
        stringlist = print_MA(A, B, 1, K)


    if fitclass == 'SMA':
        alpha = 1./params[-1]
        A = params[[i for i in range(K*2) if i % 2 != 0]]
        B = params[[i for i in range(K*2) if i % 2 == 0]]

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

    return f

if __name__ == "__main__":
    n = 51
    u = np.logspace(0, np.log10(3), n)
    w = (u**2+3) / (u+1)**2
    f = plot_fit_1d(u, w, K=2, fitclass='SMA', plotspace='linear')
