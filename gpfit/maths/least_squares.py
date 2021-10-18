"Implements the Levenberg-Marquardt algorithm"
from time import time
from sys import float_info
import numpy as np
from numpy.linalg import norm
from scipy.sparse import spdiags, issparse


# pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements,no-else-break
def levenberg_marquardt(
    residfun,
    initparams,
    verbose=False,
    lambdainit=0.02,
    maxiter=5000,
    maxtime=np.inf,
    tolgrad=np.sqrt(float_info.epsilon),
    tolrms=1e-7,
):
    """
    Levenberg-Marquardt alogrithm
    Minimizes sum of squared error of residual function

    Arguments
    ---------
    residfun: function
        Mapping from parameters to residuals, of the form
        (r, drdp) = residfun(params)
        Examples:
            if residfun is (ydata - y(params)), drdp = - dydp
            if residfun is (y(params) - ydata), drdp = dydp
    initparams: np.array (1D)
        Initial fit parameter guesses
    verbose: bool
        If true, print(verbose output)
    lambdainit: float
        Initial value for step size penalty lambda
    maxiter: int
        Maximum number of iterations before terminating
    maxtime: float
        Maximum total time (seconds) before terminating
    tolgrad: float
        First-order optimality tolerance
    tolrms: float
        Tolerance on change in rms error per iteration

    Returns
    -------
    params: np.array (1D)
        Parameter vector that locally minimizes norm(residfun, 2)
    rmstraj: np.array
        History of RMS errors after each step (first point is initialization)
    """

    t = time()

    # Check incoming params
    nparam = initparams.size

    if initparams.ndim > 1:
        raise ValueError("params should be a vector")

    # Define display formatting if required
    formatstr1 = "%5.0f        %9.6g        %9.3g\n"
    formatstr = (
        "%5.0f        %9.6g        %9.3g        %12.4g        %12.4g        %8.4g\n"
    )

    # Get residual values and jacobian at initial point; extract size info
    params = initparams
    params_updated = True
    r, J = residfun(params)  # r is a row vector, J is a Jacobian
    npt = r.size
    r.shape = (npt, 1)  # Make r into column vector

    if J.shape != (npt, nparam):
        errstr = f"Jacobian size {J.shape} inconsistent with ({npt}, {nparam})"
        raise ValueError(errstr)

    # "Accept" initial point
    rms = norm(r)/np.sqrt(npt)  # 2-norm
    maxgrad = norm(np.dot(r.T, J), ord=np.inf)  # Inf-norm
    prev_trial_accepted = False

    # Initializations
    itr = 0
    Jissparse = issparse(J)
    diagJJ = sum(J*J, 0).T
    zeropad = np.zeros((nparam, 1))
    lamb = lambdainit
    rmstraj = [rms]

    # Display info for 1st iter
    if verbose:
        print("\n                    First-Order                        " "Norm of \n")
        print(
            "Iter        Residual        optimality            Lambda"
            "            step        Jwarp \n"
        )
        print(formatstr1 % (itr, rms, maxgrad))

    # Main Loop
    while True:
        if itr == maxiter:
            if verbose:
                print("Reached maximum number of iterations")
            break
        elif time() - t > maxtime:
            if verbose:
                print(f"Reached maxtime ({maxtime} seconds)")
            break
        elif itr >= 2 and abs(rmstraj[itr] - rmstraj[itr - 2]) < rmstraj[itr]*tolrms:
            # Should really only allow this exit case
            # if trust region constraint is slack
            if verbose:
                print("RMS changed less than tolrms")
            break

        itr += 1

        # Compute diagonal scaling matrix based on curent lambda and J
        # Note this matrix changes every iteration,
        # since either lambda or J (or both) change every iteration
        if Jissparse:
            # spdiags takes the transpose of the matrix in matlab
            D = spdiags(np.sqrt(lamb*diagJJ), 0, nparam, nparam)
        else:
            D = np.diag(np.sqrt(lamb*diagJJ))

        # Update augmented least squares system
        if params_updated:
            diagJJ = sum(J*J, 0).T
            augJ = np.vstack((J, D))
            r.shape = (npt, 1)
            augr = np.vstack((-r, zeropad))
        else:
            augJ[npt:, :] = D  # modified from npt+1 ??

        # Compute step for this lambda
        step = np.linalg.lstsq(augJ, augr, rcond=-1)[0]
        # Rank condition specified to default for python upgrades
        trialp = (params + step.T)[0]

        # Check function value at trialp
        trialr, trialJ = residfun(trialp)
        trialrms = norm(trialr)/np.sqrt(npt)
        rmstraj.append(trialrms)

        # Accept or reject trial params
        if trialrms < rms:
            params = trialp
            J = trialJ
            r = trialr
            rms = trialrms
            maxgrad = norm(np.dot(r.T, J), np.inf)
            # dsp here so that all grad info is for updated point,
            # but lambda not yet updated
            if verbose:
                print(formatstr % (itr, trialrms, maxgrad, lamb, norm(step),
                                   max(diagJJ)/min(diagJJ)))

            if maxgrad < tolgrad:
                if verbose:
                    print("1st order optimality attained")
                break

            if prev_trial_accepted and itr > 1:
                lamb = lamb/10

            prev_trial_accepted = True
            params_updated = True
        else:
            if verbose:
                print(formatstr % (itr, trialrms, maxgrad, lamb, norm(step),
                                   max(diagJJ)/min(diagJJ)))
            lamb = lamb*10
            prev_trial_accepted = False
            params_updated = False

    assert len(rmstraj) == itr + 1
    rmstraj = np.array(rmstraj)
    if verbose:
        print("Final RMS: " + repr(rms))

    return params, rmstraj
