" evaluate a fitted constraint "
import numpy as np
from max_affine import max_affine
from softmax_affine import softmax_affine
from implicit_softmax_affine import implicit_softmax_affine

def evaluate_fit(cnstr, x, fittype):
    """
    given a constraint and x data, return y

    Inputs
    ------
    cnstr: Constraint - (MonomialInequality, MonomialEquality,
                         PosynomialInequality)
    x: 1D or 2D array - array of input values in log space
    fittype: string - "MA", "SMA",  or "ISMA"

    Output
    ------
    y: 1D array - array of output for the given x inputs in log space

    """

    y = 0

    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    else:
        x = x.T

    if fittype == "MA":
        if not hasattr(cnstr, "__len__"):
            cnstr = [cnstr]
        vkn = range(len(cnstr[0].varkeys))
        expos = np.array(
            [cn.left.exp[list(cn.varkeys["u_%d" % n])[0]] for cn in cnstr
             for n in vkn]).reshape(len(cnstr), len(vkn))
        params = np.hstack([np.hstack([np.log(cn.left.c), ex])
                            for cn, ex in zip(cnstr, expos)])
        if np.inf in params or 0.0 in params or -np.inf in params:
            pass
        else:
            y, _ = max_affine(x, params)

    elif fittype == "SMA":
        wvk = [vk for vk in cnstr.varkeys if vk.name == "w"][0]
        alpha = [1/ex[wvk] for ex in cnstr.left.exps][0]
        vkn = range(len(cnstr.right.varkeys))
        expos = np.array(
            [e[list(cnstr.varkeys["u_fit_(%d,)" % n])[0]] for e in cnstr.right.exps
             for n in vkn]).reshape(len(cnstr.right.cs), len(vkn))
        params = np.hstack([np.hstack([np.log(c**(alpha))] + [ex*alpha])
                            for c, ex in zip(cnstr.right.cs, expos)])
        params = np.append(params, alpha)
        if np.inf in params or 0.0 in params or -np.inf in params:
            pass
        else:
            y, _ = softmax_affine(x, params)

    elif fittype == "ISMA":
        wvk = [vk for vk in cnstr.varkeys if vk.name == "w"][0]
        alphas = [-1/ex[wvk] for ex in cnstr.left.exps]
        vkn = range(1, len(cnstr.varkeys))
        expos = np.array(
            [e[list(cnstr.varkeys["u_%d" % n])[0]] for e in cnstr.left.exps
             for n in vkn]).reshape(len(cnstr.left.cs), len(vkn))
        params = np.hstack([np.hstack([np.log(c**a)] + [e*a]) for c, e, a in
                            zip(cnstr.left.cs, expos, alphas)])
        params = np.append(params, alphas)
        if np.inf in params or 0.0 in params or -np.inf in params:
            pass
        else:
            y, _ = implicit_softmax_affine(x, params)

    return y
