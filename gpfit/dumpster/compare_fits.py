from max_affine_init import max_affine_init
from max_affine import max_affine
from softmax_affine import softmax_affine
from implicit_softmax_affine import implicit_softmax_affine
from time import time
from LM import LM
from generic_resid_fun import generic_resid_fun
from numpy import append, ones, size


def compare_fits(xdata, ydata, Ks, ntry):
    '''
    INPUTS
        xdata: multi-dimensional x data (independent variable)
                [# data points, # dimensions]
                e.g. [[1,2],[8,7],[34,6]] for three 2-dimensional data points

        ydata: one-dimensional y data (dependent variable)
                [# data points, 1]
                e.g. [5,77,-4] for three data points

        Ks: 1D array containing values of k,
            where k is the number of monomials in posynomial approx.
            [# values of k, 1]
            Can be either numpy array of integers or an integer
            e.g. [1, 3] for a monomial and a 3-monomial posynomial expression
                Values of k must be integers

        ntry: Number of tries (related to number of starts).
              Higher is better but more expensive.
                e.g. 10, for 10 tries
                Value must be integer

    OUTPUTS
        s:        Dictionary containing the results of the 4 different methods
    '''
    def store_results(fieldname):
        s[fieldname]['resid'][ik][tt] = min(RMStraj)
        s[fieldname]['iter'][ik][tt] = len(RMStraj)
        s[fieldname]['params'][ik][tt] = params
        s[fieldname]['time'][ik][tt] = time() - t

    s = AutoVivification()  # Clean, easy way to initialize nested-dictionaries
    s['Ks'] = Ks
    alphainit = 10

    for ik in range(size(Ks)):
        for tt in range(ntry):

            if size(Ks) > 1:
                k = Ks[ik]
            else:
                k = Ks

            bainit = max_affine_init(xdata, ydata, k)

            def rfun(p):
                return generic_resid_fun(max_affine, xdata, ydata, p)
            t = time()
            # 'F' flattens in 'fortran' order i.e. column-major
            [params, RMStraj] = LM(rfun, bainit.flatten('F'))
            store_results('max_affine')

            def rfun(p):
                return generic_resid_fun(softmax_affine, xdata, ydata, p)
            t = time()
            [params, RMStraj] = LM(rfun,
                                   append(params.flatten('F'), alphainit))
            store_results('softmax_optMAinit')

            def rfun(p):
                return generic_resid_fun(softmax_affine, xdata, ydata, p)
            t = time()
            [params, RMStraj] = LM(rfun,
                                   append(bainit.flatten('F'), alphainit))
            store_results('softmax_originit')

            def rfun(p):
                return generic_resid_fun(implicit_softmax_affine,
                                         xdata, ydata, p)
            t = time()
            [params, RMStraj] = LM(rfun,
                                   append(bainit.flatten('F'),
                                          alphainit*ones((k, 1))))
            store_results('implicit_originit')

    return s


# Autovivification allows for easy and clean initiation of nested dictionaries
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
