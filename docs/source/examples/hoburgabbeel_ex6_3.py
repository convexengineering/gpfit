"Example 6.3 from Hoburg/Abbeel GPfit paper"
import numpy as np
from numpy.random import seed, random_sample
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

seed(33404)

Vdd = random_sample(1000) + 1
Vth = 0.2*random_sample(1000) + 0.2
P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)
u = np.vstack((Vdd, Vth))
x = np.log(u)
y = np.log(P)
K = 4

fma = MaxAffine(x, y, K)
fsma = SoftmaxAffine(x, y, K)
fisma = ImplicitSoftmaxAffine(x, y, K)

print("MA RMS Error: %.5g" % fma.error["rms"])
print("SMA RMS Error: %.5g" % fsma.error["rms"])
print("ISMA RMS Error: %.5g" % fisma.error["rms"])
