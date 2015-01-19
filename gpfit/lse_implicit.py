from numpy import zeros, ones, spacing, exp, log, tile

def lse_implicit(x,alpha):        
    '''
    Implicit Log-sum-exponential function with derivatives
    - sums across the second dimension of x
    - returns one y for every row of x
    - lse_implicit is a mapping R^n --> R, where n==size(x,2)
    - implementation: newton raphson steps to find f(x,y) = 0

    INPUTS:
            x:      independent variable data?
                    [n x ? 2D array], n is number of data points

            alpha:  local softness parameter for each column of x
                    [1D array with size(alpha)==size(x,2)]

    OUTPUTS:
            y:      dependent variable fit. Returns one y for every row of x.
                    [n-element 1D array], n is number of data points

            dydx:   dydx gives the deriv of each y wrt each x 
                    [n x ? 2D array]

            dydalpha:
                    [n x ? 2D array], n is number of data points

    '''
    bverbose = False

    tol = 10*spacing(1) 
    npt,nx = x.shape

    if not alpha.size == nx:
        raise Exception('alpha size mismatch')

    alphamat = tile(alpha, (npt,1))

    m = x.max(1)  #maximal x values

    h = x - (tile(m, (nx, 1))).T   #distance from m; note h <= 0 for all entries
    #(unless x has infs; in this case h has nans; can prob deal w/ this gracefully)
    #should also deal with alpha==inf case gracefully

    L = zeros((npt,))  #initial guess. note y = m + L

    Lmat = (tile(L,(nx,1))).T

    #initial eval
    expo = exp(alphamat*(h-Lmat))
    alphaexpo = alphamat*expo
    sumexpo = expo.sum(axis=1) #<<<<<<<<<<<check that lack of transpose isnt issue<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    sumalphaexpo = alphaexpo.sum(axis=1)
    f = log(sumexpo)
    dfdL = -alphaexpo.sum(axis=1)/sumexpo
    neval = 1
    i = abs(f) > tol #inds to update
    #disp(['max newton-raphson residual: ', num2str(max(abs(f)))]);

    while any(i):
        L[i] = L[i] - f[i]/dfdL[i]    #newton step
        #re-evaluate
        Lmat[i,:] = (tile(L[i],(nx,1))).T
        expo[i,:] = exp(alphamat[i,:] * (h[i,:] - Lmat[i,:]))
        alphaexpo[i,:] = alphamat[i,:] * expo[i,:]
        sumexpo[i] = expo[i,:].sum(axis=1)
        sumalphaexpo[i,] = alphaexpo[i,:].sum(axis=1) # sumalphaexpo[i,:] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        f[i] = log(sumexpo[i])
        dfdL[i] = -sumalphaexpo[i,]/sumexpo[i]
        neval = neval + 1
        if neval > 40:
            print('') #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ask woody

        #update inds that need to be evaluated
        i[i] = abs(f[i]) > tol
        #disp(['max newton-raphson residual: ', num2str(max(abs(f)))]);

    if bverbose:
        print('lse_implicit converged in ' + repr(neval) + ' newton-raphson steps')

    y = m + L

    dydx = alphaexpo/(tile(sumalphaexpo, (nx,1))).T
    dydalpha = (h - Lmat)*expo/(tile(sumalphaexpo, (nx,1))).T

    return y, dydx, dydalpha