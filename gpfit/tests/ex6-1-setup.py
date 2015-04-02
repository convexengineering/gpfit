from numpy import linspace, logspace, log, exp, log10
from fit import fit

m = 501

u = logspace(0,log10(3),501)
#u = u.reshape(u.size,1)
w = (u**2 + 3)/(u+1)**2
#w = w.reshape(w.size,1)

x = log(u)
y = log(w)