from numpy import arange, meshgrid, hstack, linspace, log, exp, maximum, sqrt, mean, square
from gpfit.compare_fits import compare_fits
from gpfit.implicit_softmax_affine import implicit_softmax_affine

def albatross():
    [X, Y] = meshgrid(arange(1.,6.),arange(1.,6.))
    Z = X**2+Y**2
    K = 2
    compute_fit(X,Y,Z,K)

def alligator():
    [X, Y] = meshgrid(arange(1.,6.),arange(1.,6.))
    Z = (X**2+Y**2)**0.5
    K = 2
    compute_fit(X,Y,Z,K)

def angelfish():
    [X, Y] = meshgrid(arange(1.,6.),arange(1.,6.))
    Z = X/Y + Y/X
    K = 2
    compute_fit(X,Y,Z,K)

def anteater():
    [X, Y] = meshgrid(arange(1.,6.),arange(1.,6.))
    Z = (X/Y + Y/X)**0.5
    K = 2
    compute_fit(X,Y,Z,K)

def antelope():
    [X, Y] = meshgrid(linspace(1.,2.,30),linspace(0.2,0.4,30))
    Z = (1.09*X**4.27*Y**0.112 + (7.79e-5)*X**4.75/Y**6.44 + (1.36e-7)*X**8.94/Y**8.89)**(1/2.14)
    K = 2
    compute_fit(X,Y,Z,K)

def appenzeller():
    [X, Y] = meshgrid(linspace(1.,2.,30),linspace(0.2,0.4,30))
    Z = X**2 + 30*X*exp(-(Y-0.06*X)/0.039)
    K = 2
    compute_fit(X,Y,Z,K)

def armadillo():
    [X, Y] = meshgrid(linspace(1.,2.,30),linspace(0.2,0.4,30))
    Z = X**2 + 30*X*exp(-(Y-0.06*X)/0.039)
    K = 3
    compute_fit(X,Y,Z,K)

def compute_fit(X,Y,Z,K):
    u1, u2 = X, Y

    w = Z.reshape(Z.size,1)
    X = X.reshape(X.size,1)
    Y = Y.reshape(Y.size,1)
    u = hstack((X,Y))

    x = log(u)
    y = log(w)

    s = compare_fits(x,y, K,1)

    PAR_MA = s['max_affine']['params'][0][0]
    PAR_SMA = s['softmax_optMAinit']['params'][0][0]
    PAR_ISMA = s['implicit_originit']['params'][0][0]

    if K == 2:
        # Max Affine Fitting
        A = PAR_MA[[1,2,4,5]]
        B = PAR_MA[[0,3]]
        w_MA_1 = exp(B[0]) * u1**A[0] * u2**A[1]
        w_MA_2 = exp(B[1]) * u1**A[2] * u2**A[3]
        w_MA = maximum(w_MA_1, w_MA_2)

        # Softmax Affine Fitting        
        A = PAR_SMA[[1,2,4,5]]
        B = PAR_SMA[[0,3]]
        alpha = 1.0/PAR_SMA[-1]
        w_SMA = (exp(alpha*B[0]) * u1**(alpha*A[0]) * u2**(alpha*A[1]) +
                exp(alpha*B[1]) * u1**(alpha*A[2]) * u2**(alpha*A[3])
                )**(1.0/alpha)

    elif K ==3:
        A = PAR_MA[[1,2,4,5,7,8]]
        B = PAR_MA[[0,3,6]]
        w_MA_1 = exp(B[0]) * u1**A[0] * u2**A[1]
        w_MA_2 = exp(B[1]) * u1**A[2] * u2**A[3]
        w_MA_3 = exp(B[2]) * u1**A[3] * u2**A[4]
        w_MA = maximum(w_MA_1, w_MA_2, w_MA_3)

        A = PAR_SMA[[1,2,4,5,7,8]]
        B = PAR_SMA[[0,3,6]]
        alpha = 1.0/PAR_SMA[-1]
        w_SMA = (exp(alpha*B[0]) * u1**(alpha*A[0]) * u2**(alpha*A[1]) +
                exp(alpha*B[1]) * u1**(alpha*A[2]) * u2**(alpha*A[3]) +
                exp(alpha*B[2]) * u1**(alpha*A[4]) * u2**(alpha*A[5])
        )**(1.0/alpha)

    MA_rms_pct_error = sqrt(mean(square((w_MA - Z)/Z)))
    print MA_rms_pct_error
    SMA_rms_pct_error = sqrt(mean(square((w_SMA - Z)/Z)))
    print SMA_rms_pct_error

    # Implicit Softmax Affine Fitting
    w_ISMA, _ = implicit_softmax_affine(x,PAR_ISMA)
    w_ISMA = exp(w_ISMA)
    ISMA_rms_pct_error = sqrt(mean(square((w_ISMA - w.T[0])/w.T[0])))
    print ISMA_rms_pct_error

