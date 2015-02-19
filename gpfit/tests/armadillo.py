from numpy import arange, meshgrid, hstack, log, exp, amax, maximum, linspace
from gpfit.compare_fits import compare_fits

[X, Y] = meshgrid(linspace(1,2,30),linspace(0.2,0.4,30))
Z = X**2 + 30*X*exp(-(Y-0.06*X)/0.039)

u1, u2 = X, Y

w = Z.reshape(Z.size,1)
X = X.reshape(X.size,1)
Y = Y.reshape(Y.size,1)
u = hstack((X,Y))

x = log(u)
y = log(w)

s = compare_fits(x,y, 3,1)


# Max Affine Fitting
PAR_MA = s['max_affine']['params'][0][0]
A = PAR_MA[[1,2,4,5,7,8]]
B = PAR_MA[[0,3,6]]
w_MA_1 = exp(B[0]) * u1**A[0] * u2**A[1]
w_MA_2 = exp(B[1]) * u1**A[2] * u2**A[3]
w_MA_3 = exp(B[2]) * u1**A[3] * u2**A[4]
w_MA = maximum(w_MA_1, w_MA_2, w_MA_3)
print amax(w_MA - Z)


# Softmax Affine Fitting
PAR_SMA = s['softmax_optMAinit']['params'][0][0]
A = PAR_SMA[[1,2,4,5,7,8]]
B = PAR_SMA[[0,3,6]]
alpha = 1/PAR_SMA[-1]
w_SMA = (exp(alpha*B[0]) * u1**(alpha*A[0]) * u2**(alpha*A[1]) +
        exp(alpha*B[1]) * u1**(alpha*A[2]) * u2**(alpha*A[3]) +
        exp(alpha*B[2]) * u1**(alpha*A[4]) * u2**(alpha*A[5])
        )**(1/alpha)
print amax(w_SMA - Z)

# Implicit Softmax Affine Fitting
PAR_ISMA = s['implicit_originit']['params'][0][0]
A = PAR_ISMA[[1,2,4,5,7,8]]
B = PAR_ISMA[[0,3,6]]
alpha = 1/PAR_ISMA[[-3, -2,-1]]


def implicit_function(x,y,z):
	return (exp(alpha[0]*B[0])/z**alpha[0] * x**(alpha[0]*A[0]) * y**(alpha[0]*A[1]) +
     		exp(alpha[1]*B[1])/z**alpha[1] * x**(alpha[1]*A[2]) * y**(alpha[1]*A[3]) +
     		exp(alpha[2]*B[2])/z**alpha[2] * x**(alpha[2]*A[4]) * y**(alpha[2]*A[5]) - 1
    		)
