"""Example 6.1 from Hoburg et al."""
import numpy as np
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

u = np.logspace(0, np.log10(3), 101)
w = (u**2 + 3)/(u + 1)**2
x = np.log(u)
y = np.log(w)
K = 3

fma = MaxAffine(x, y, K, verbosity=1)
fsma = SoftmaxAffine(x, y, K, verbosity=1)
fisma = ImplicitSoftmaxAffine(x, y, K, verbosity=1)
