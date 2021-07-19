"Fits an example function"
from numpy import logspace, log, log10, random
from gpfit.fit import MaxAffine, SoftmaxAffine, ImplicitSoftmaxAffine

random.seed(33404)

u = logspace(0, log10(3), 101)
w = (u**2 + 3)/(u + 1)**2
x = log(u)
y = log(w)
K = 3

fma = MaxAffine(x, y, K)
fsma = SoftmaxAffine(x, y, K)
fisma = ImplicitSoftmaxAffine(x, y, K)

print("MA RMS Error: %.5g" % fma.error["rms"])
print("SMA RMS Error: %.5g" % fsma.error["rms"])
print("ISMA RMS Error: %.5g" % fisma.error["rms"])
