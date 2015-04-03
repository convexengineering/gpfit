from numpy import sqrt, shape, zeros, vstack, dot, ndim, inf, diag
from numpy.linalg import inv, norm, lstsq
from scipy.sparse import spdiags, issparse
from time import time
from sys import float_info


def LM(residfun, initparams):
    '''
    Levenber-Marquardt alogrithm
    Minimizes sum of squared error of residfun(params)

    INPUTS:
        residfun:     [r, drpdp] = residfun(params)
                        if residfun is (ydata - y(params)), drdp = - dydp
                        if residfun is (y(params) - ydata), drdp = dydp

        params:     column vector of initial param guesses

    OUTPUTS:
        params:     best params found

        RMStraj:     history of RMS errors after each step
                        first point is initialization
    '''

    t = time()

    # Check incoming params
    nparam = initparams.size

    if initparams.ndim > 1:
        raise Exception('params should be a vector')

    # Set defaults; incorporate incoming options
    defaults = {}
    defaults['bverbose'] = False
    # defaults['bplot'] = True
    defaults['lambdainit'] = 0.02
    defaults['maxiter'] = 5000
    defaults['maxtime'] = 5
    defaults['tolgrad'] = sqrt(float_info.epsilon)
    defaults['tolrms'] = 1E-7
    options = defaults

    # Define display formatting if required
    formatstr1 = '%5.0f        %9.6g        %9.3g\n'
    formatstr = ('%5.0f        %9.6g        %9.3g        '
                 '%12.4g        %12.4g        %8.4g\n')

    # Get residual values and jacobian at initial point; extract size info
    params = initparams
    params_updated = True
    r, J = residfun(params)  # r is a row vector, J is a Jacobian
    npt = r.size
    r.shape = (npt, 1)  # Make r into column vector

    if J.shape != (npt, nparam):
        raise Exception('Jacobian size inconsistent')

    # "Accept" initial point
    rms = norm(r)/sqrt(npt)  # 2-norm
    maxgrad = norm(dot(r.T, J), ord=inf)  # Inf-norm
    prev_trial_accepted = False

    # Initializations
    itr = 0
    Jissparse = issparse(J)
    diagJJ = sum(J*J, 0).T  # <<< why not use diag()?, what does ( ,1) do?
    zeropad = zeros((nparam, 1))
    lamb = options['lambdainit']
    RMStraj = zeros((options['maxiter'] + 1, 1))
    RMStraj[0] = rms
    gradcutoff = options['tolgrad']

    # Display info for 1st iter
    if options['bverbose']:
        print('\n                    First-Order                        '
              'Norm of \n')
        print('Iter        Residual        optimality            Lambda'
              '            step        Jwarp \n')
        print formatstr1 % (itr, rms, maxgrad)

    # Main Loop
    while True:

        if itr == options['maxiter']:
            if options['bverbose']:
                print 'Reached maximum number of iterations'
            break

        elif time() - t > options['maxtime']:
            if options['bverbose']:
                print('Reached maxtime (' +
                      repr(options['maxtime']) +
                      ' seconds)')
            break
        elif (itr >= 2 and
              abs(RMStraj[itr] - RMStraj[itr-2]) <
              RMStraj[itr]*options['tolrms']):
            # Should really only allow this exit case
            # if trust region constraint is slack
            if options['bverbose']:
                print('RMS changed less than tolrms')
            break

        itr += 1

        # Compute diagonal scaling matrix based on curent lambda and J
        # Note this matrix changes every iteration,
        # since either lambda or J (or both) change every iteration
        if Jissparse:
            # spdiags takes the transpose of the matrix in matlab
            D = spdiags(sqrt(lamb*diagJJ), 0, nparam, nparam)
            print 'is it? is it??'
        else:
            D = diag(sqrt(lamb*diagJJ))

        # Update augmented least squares system
        if params_updated:
            diagJJ = sum(J*J, 0).T
            augJ = vstack((J, D))
            r.shape = (npt, 1)
            augr = vstack((-r, zeropad))
        else:
            augJ[npt:, :] = D  # modified from npt+1 ??

        # Compute step for this lambda
        step = lstsq(augJ, augr)[0]
        trialp = (params + step.T)[0]

        # Check function value at trialp
        trialr, trialJ = residfun(trialp)
        trialrms = norm(trialr)/sqrt(npt)
        RMStraj[itr] = trialrms

        # Accept or reject trial params
        if trialrms < rms:
            params = trialp
            J = trialJ
            r = trialr
            rms = trialrms
            maxgrad = norm(dot(r.T, J), inf)
            # dsp here so that all grad info is for updated point,
            # but lambda not yet updated
            if options['bverbose']:
                print formatstr % (itr, trialrms, maxgrad, lamb, norm(step),
                                   max(diagJJ)/min(diagJJ))

            if maxgrad < gradcutoff:
                if options['bverbose']:
                    print('1st order optimality attained')
                break

            if prev_trial_accepted and itr > 1:
                lamb = lamb/10

            prev_trial_accepted = True
            params_updated = True
        else:
            if options['bverbose']:
                print formatstr % (itr, trialrms, maxgrad, lamb, norm(step),
                                   max(diagJJ)/min(diagJJ))
            lamb = lamb*10
            prev_trial_accepted = False
            params_updated = False

    RMStraj = RMStraj[1:itr+1]
    if options['bverbose']:
        print('Final RMS: ' + repr(rms))

    return params, RMStraj
