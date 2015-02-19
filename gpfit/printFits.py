from numpy import arange, exp, array

def print_max_affine(PAR_MA, K):
	'''
	Print set of K monomial inequality constraints
	'''
	d = PAR_MA.size/K - 1 #Number of dimensions (independent variables)

	A = PAR_MA[[i for i in range(PAR_MA.size) if i % (N + 1) != 0]]
	B = PAR_MA[[i for i in range(PAR_MA.size) if i % (N + 1) == 0]]

	for k in range(K):
		printString = 'w = {0:.3g}'.format(exp(B[k]))
		
		for i in range(d):
			printString += ' * (u_{0:d})**{1:.3g}'.format(i, A[d*k + i])

		print printString


def print_softmax_affine(PAR_SMA, K):

	d = (PAR_SMA.size - 1)/K - 1 #Number of independent dimensions (independent variables)

	alpha = 1/PAR_SMA[-1]
	A = PAR_SMA[[i for i in range(PAR_SMA.size) if i % (d + 1) != 0]]
	B = PAR_SMA[[i for i in range(PAR_SMA.size) if i % (d + 1) == 0]]

	printString = 'w**{0:.3g} = '.format(alpha)
	for k in range(K):
		if k != 0:
			printString += ' + '

		printString += '{0:.3g}'.format(exp(alpha * B[k]))
		
		for i in range(d):
			printString += ' * (u_{0:d})**{1:.3g}'.format(i, alpha * A[d*k + i])

	print printString
	

def print_implicit_softmax_affine(PAR_ISMA, K):
