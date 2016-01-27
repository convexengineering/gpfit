from gpfit.fit import fit
from numpy import logspace, log, log10

n = 501
u = logspace(0,log10(3),n)
w = (u**2+3)/(u+1)**2
x = log(u)
y = log(w)
K = 3
Type = "ISMA"

cstrt, rms_error = fit(x,y,K,Type)
