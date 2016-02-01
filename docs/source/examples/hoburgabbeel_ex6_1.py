from gpfit.fit import fit
from numpy import logspace, log, log10

n = 501
u = logspace(0,log10(3),n)
w = (u**2+3)/(u+1)**2
x = log(u)
y = log(w)
K = 3

cMA, errorMA = fit(x,y,K,"MA")
cSMA, errorSMA = fit(x,y,K,"SMA")
cISMA, errorISMA = fit(x,y,K,"ISMA")

assert errorMA < 1e-2
assert errorSMA < 1e-4
assert errorISMA < 1e-5

print "MA RMS Error: ", errorMA
print "SMA RMS Error: ", errorSMA
print "ISMA RMS Error: ", errorISMA
