from numpy import arange, exp, array

def print_MA(PAR_MA, K):
	'''
	Print set of K monomial inequality constraints
	'''
	d = PAR_MA.size/K - 1 #Number of dimensions (independent variables)

	A = PAR_MA[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
	B = PAR_MA[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]
	stringList = [None]*K

	for k in range(K):
		printString = 'w = {0:.3g}'.format(exp(B[k]))
		
		for i in range(d):
			printString += ' * (u_{0:d})**{1:.3g}'.format(i, A[d*k + i])

		stringList[k] = printString
		print printString
		
	return stringList


def print_SMA(PAR_SMA, K):

	d = (PAR_SMA.size - 1)/K - 1 #Number of independent dimensions (independent variables)

	alpha = 1./PAR_SMA[-1]

	A = PAR_SMA[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
	B = PAR_SMA[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

	stringList = [None]*K

	printString = 'w**{0:.3g} = '.format(alpha)
	for k in range(K):
		if k > 0:
			print printString
			printString = '    + '

		printString += '{0:.3g}'.format(exp(alpha * B[k]))
		
		for i in range(d):
			printString += ' * (u_{0:d})**{1:.3g}'.format(i, alpha * A[d*k + i])

		stringList[k] = printString

	print printString
	return stringList
	

def print_ISMA(PAR_ISMA, K):

	d = (PAR_ISMA.size - K)/K - 1 #Number of independent dimensions (independent variables)

	alpha = 1./PAR_ISMA[range(-K,0)]

	A = PAR_ISMA[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
	B = PAR_ISMA[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

	stringList = [None]*K

	printString = '1 = '
	for k in range(K):
		if k > 0:
			print printString
			printString = '    + '

		printString += '({0:.3g}/w**{1:.3g})'.format(exp(alpha[k] * B[k]), alpha[k])
		
		for i in range(d):
			printString += ' * (u_{0:d})**{1:.3g}'.format(i, alpha[k] * A[d*k + i])

		stringList[k] = printString

	print printString
	return stringList

