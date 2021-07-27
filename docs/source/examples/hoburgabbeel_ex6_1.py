"Fits an example function"
from numpy import logspace, log, log10, random
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

random.seed(33404)

u = logspace(0, log10(3), 101)
w = (u**2 + 3)/(u + 1)**2
x = log(u)
y = log(w)
K = 3

fma = MaxAffine(x, y, K, verbosity=1)
fsma = SoftmaxAffine(x, y, K, verbosity=1)
fisma = ImplicitSoftmaxAffine(x, y, K, verbosity=1)
