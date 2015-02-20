from numpy import arange, meshgrid, hstack, log, exp, maximum, sqrt, mean, square
from gpfit.compare_fits import compare_fits
from gpfit.implicit_softmax_affine import implicit_softmax_affine

[X, Y] = meshgrid(arange(1.,6.),arange(1.,6.))
Z = X/Y + Y/X

u1, u2 = X, Y

w = Z.reshape(Z.size,1)
X = X.reshape(X.size,1)
Y = Y.reshape(Y.size,1)
u = hstack((X,Y))

x = log(u)
y = log(w)

s = compare_fits(x,y, 2,1)


# Max Affine Fitting
PAR_MA = s['max_affine']['params'][0][0]
A = PAR_MA[[1,2,4,5]]
B = PAR_MA[[0,3]]
w_MA_1 = exp(B[0]) * u1**A[0] * u2**A[1]
w_MA_2 = exp(B[1]) * u1**A[2] * u2**A[3]
w_MA = maximum(w_MA_1, w_MA_2)
MA_rms_error = sqrt(mean(square(w_MA - Z)))
print MA_rms_error


# Softmax Affine Fitting
PAR_SMA = s['softmax_optMAinit']['params'][0][0]
A = PAR_SMA[[1,2,4,5]]
B = PAR_SMA[[0,3]]
alpha = 1.0/PAR_SMA[-1]
w_SMA = (exp(alpha*B[0]) * u1**(alpha*A[0]) * u2**(alpha*A[1]) +
        exp(alpha*B[1]) * u1**(alpha*A[2]) * u2**(alpha*A[3])
        )**(1.0/alpha)

SMA_rms_error = sqrt(mean(square(w_SMA - Z)))
print SMA_rms_error

# Implicit Softmax Affine Fitting
PAR_ISMA = s['implicit_originit']['params'][0][0]
w_ISMA, _ = implicit_softmax_affine(x,PAR_ISMA)
w_ISMA = exp(w_ISMA)
ISMA_rms_error = sqrt(mean(square(w_ISMA - w.T[0])))
print ISMA_rms_error