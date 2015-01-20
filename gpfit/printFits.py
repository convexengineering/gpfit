from numpy import arange, exp, array

def print_max_affine(params, k):
	'''
	Print set of K monomial inequality constraints
	'''
	b_ind = 2*(arange(1,k+1) - 1)
	a_ind = 2*arange(1,k+1) - 1

	a = params[:,a_ind]
	b = params[:,b_ind]
	
	for ik in range(k):
		printstr = 'w >= {0:.3g}'.format(exp(b[0,ik]))

		for ii in range(params.shape[0]):
			printstr += '* u_{0}^{1:.3g}'.format(ii, a[ii,ik])

		print printstr

	return a, b

def print_softmax_affine():
	pass

def print_implicit_softmax_affine():
	pass


params = array([[0.,1.,-10.,2.]])
k = 2
print_max_affine(params,k)