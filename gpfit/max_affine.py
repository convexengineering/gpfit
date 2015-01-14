from numpy import zeros, hstack, reshape, ones, dot, arange, equal, ix_

def max_affine(x,ba,flag=False):
	'''
	ba may come in as a matrix or as a vector
	after reshaping (column-major), ba is dimx+1 by K
	first row is b
	rest is a	
	'''	
	npt, dimx = x.shape
	K = ba.size/(dimx+1)
	ba = reshape(ba,(dimx+1,K), order='F') # 'F' gives Fortran order indexing
	
	#augment data with column of ones
	X = hstack((ones((npt, 1)), x))

	y, partition = (dot(X,ba)).max(1), (dot(X,ba)).argmax(1)

	#The not-sparse sparse version
	dydba = zeros((npt,(dimx+1)*K))
	for k in range(K):
		inds = equal(partition, k)
		indadd = (dimx+1)*(k) 
		ixgrid = ix_(inds.nonzero()[0],indadd + arange(dimx+1))
		dydba[ixgrid] = X[inds,:]

	return y, dydba