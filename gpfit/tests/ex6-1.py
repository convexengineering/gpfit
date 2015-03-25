from gpfit.compare_fits import compare_fits
from gpfit.implicit_softmax_affine import implicit_softmax_affine
from numpy import linspace, logspace, log, exp, log10, sqrt, square, mean
import matplotlib.pyplot as plt

m = 501

u = linspace(1,3,m)
#u = logspace(0,log10(3),501)
u = u.reshape(u.size,1)
w = (u**2 + 3)/(u+1)**2
w = w.reshape(w.size,1)


x = log(u)
y = log(w)

s = compare_fits(x,y, 2,1)
print(s['softmax_originit']['resid'][0][0])

PAR_SMA = s['softmax_originit']['params'][0][0]

A = PAR_SMA[[1,3]]
B = PAR_SMA[[0,2]]
alpha = 1.0/PAR_SMA[-1]
w_SMA = (exp(alpha*B[0]) * u**(alpha*A[0]) +
         exp(alpha*B[1]) * u**(alpha*A[1])
        )**(1.0/alpha)
plt.plot(u,w_SMA)

SMArms = sqrt(mean(square(w_SMA-w)))
print SMArms


print(s['implicit_originit']['resid'][0][0])
PAR_ISMA = s['implicit_originit']['params'][0][0]

w_ISMA, _ = implicit_softmax_affine(x,PAR_ISMA)
w_ISMA = exp(w_ISMA)

plt.figure()
plt.plot(u,w_ISMA)

w_ISMA = w_ISMA.reshape(w_ISMA.size,1)
ISMArms = sqrt(mean(square(w_ISMA-w)))
print ISMArms

plt.show()