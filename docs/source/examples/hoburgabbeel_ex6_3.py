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

fma = MaxAffine(x, y, K, verbosity=1)
fsma = SoftmaxAffine(x, y, K, verbosity=1)
fisma = ImplicitSoftmaxAffine(x, y, K, verbosity=1)
