from numpy import arange, exp, array

def print_max_affine(params):
	'''
	Print set of k monomial inequality constraints
	'''
	K = params.shape[1]/2

	b_ind = 2*(arange(1,K+1) - 1)
	a_ind = 2*arange(1,K+1) - 1

	a = params[:,a_ind]
	b = params[:,b_ind]
	
	for k in range(K):
		printstr = 'w >= {0:.3g}'.format(exp(b[0,k]))

		for i in range(params.shape[0]):
			printstr += '* u_{0}^{1:.3g}'.format(i, a[i,k])

		print printstr

	return a, b


def print_softmax_affine(params):

	K = (params.shape[1] - 1)/2

	b_ind = 2*(arange(1,K+1) - 1)
	a_ind = 2*arange(1,K+1) - 1

	a = params[:,a_ind]
	b = params[:,b_ind]
	alpha = params[:,-1]

	printstr = 'w^{0:.3g} >= '.format(alpha)

	for k in range(K):
		if k != 1:
			printstr += ' + '

		printstr += '{0:.3g}'.format(exp(alpha * b[0,k]))

		for i in range(params.shape[0]):
			printstr += '* u_{0}^{1:.3g}'.format(i, alpha*a[i,k])

	print printstr

	return a, b, alpha
	

def print_implicit_softmax_affine():

	K = params.shape[1]/3

	b_ind = 2*(arange(1,K+1) - 1)
	a_ind = 2*arange(1,K+1) - 1

	a = params[:,a_ind]
	b = params[:,b_ind]
	alpha = params[:,-K:]

	printstr = '1 >= '

	for k in range(K):
		if k != 1:
			printstr += ' + '

		printstr += '({0:.3g} * w^-{1:.3g}'.format(exp(alpha[k] * b[0,k]), alpha[k])

		for i in range(params.shape[0]):
			printstr += '* u_{0}^{1:.3g}'.format(i, alpha[k]*a[i,k])

	print printstr

	return a, b, alpha


params = array([[0.,1.,-10.,2.]])
print_max_affine(params)