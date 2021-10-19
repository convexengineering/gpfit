"""Example 6.3 from Hoburg et al."""
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

SEED = 33404

rng = np.random.RandomState(SEED)
Vdd = rng.random_sample(1000) + 1
Vth = 0.2*rng.random_sample(1000) + 0.2
P = Vdd**2 + 30*Vdd*np.exp(-(Vth - 0.06*Vdd)/0.039)

u = np.vstack((Vdd, Vth))
w = P
x = np.log(u)
y = np.log(P)
K = 4

fma = MaxAffine(x, y, K, verbosity=1, seed=SEED)
fsma = SoftmaxAffine(x, y, K, verbosity=1, seed=SEED)
fisma = ImplicitSoftmaxAffine(x, y, K, verbosity=1, seed=SEED)
