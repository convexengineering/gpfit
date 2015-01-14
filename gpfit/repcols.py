from numpy import reshape, tile

def repcols(matin, n):
	'''
	replicate columns of a matrix
	returns a matrix with n times as many columns as matin
	example: if matin is [a b c], repcols(matin, 2) returns [a a b b c c].
	
	*Does not yet work for 1D arrays*
	'''
	nrow, ncol = matin.shape

	matout = tile(matin, (n, 1)).reshape(nrow, ncol*n, order='F')

	return matout