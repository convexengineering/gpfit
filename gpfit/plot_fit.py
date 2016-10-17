from gpfit.fit import fit
from gpfit.print_fit import print_MA, print_SMA
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_1d(x, y, K=1, fitclass='MA'):
    "Finds and plots a fit (MA or SMA) for 1D data"

    u = np.log(x)
    w = np.log(y)

    params, _ = fit(u, w, K, fitclass)

    xx = np.linspace(min(x), max(x), 100)
    if fitclass == 'MA':
        A = params[[i for i in range(K*2) if i % 2 != 0]]
        B = params[[i for i in range(K*2) if i % 2 == 0]]
        YY = []
        for k in range(K):
            YY += [np.exp(B[k])*xx**A[k]]
        stringlist = print_MA(A, B, 1, K)

    f, ax = plt.subplots()
    ax.plot(x, y, '+r')
    for yy in YY:
        ax.plot(xx, yy)
    ax.set_xlabel('x')
    ax.legend(['Data'] + stringlist, 
               loc='best')
    plt.show() 

    return f
