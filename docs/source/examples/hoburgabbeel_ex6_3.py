"Example 6.3 from Hoburg/Abbeel GPfit paper"
import numpy as np
from numpy.random import seed, random_sample
from gpfit.fit import fit

seed(33404)

Vdd = random_sample(1000) + 1
Vth = 0.2 * random_sample(1000) + 0.2
P = Vdd ** 2 + 30 * Vdd * np.exp(-(Vth - 0.06 * Vdd) / 0.039)
u = np.vstack((Vdd, Vth))
x = np.log(u)
y = np.log(P)
K = 4

_, errorMA = fit(x, y, K, "MA")
_, errorSMA = fit(x, y, K, "SMA")
_, errorISMA = fit(x, y, K, "ISMA")

print("MA RMS Error: %.5g" % errorMA)
print("SMA RMS Error: %.5g" % errorSMA)
print("ISMA RMS Error: %.5g" % errorISMA)
