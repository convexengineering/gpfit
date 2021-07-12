"Fits an example function"
from numpy import logspace, log, log10, random
from gpfit.fit import fit

random.seed(33404)

u = logspace(0, log10(3), 101)
w = (u ** 2 + 3) / (u + 1) ** 2
x = log(u)
y = log(w)
K = 3

cMA, errorMA = fit(x, y, K, "MA")
cSMA, errorSMA = fit(x, y, K, "SMA")
cISMA, errorISMA = fit(x, y, K, "ISMA")

print("MA RMS Error: %.5g" % errorMA)
print("SMA RMS Error: %.5g" % errorSMA)
print("ISMA RMS Error: %.5g" % errorISMA)
