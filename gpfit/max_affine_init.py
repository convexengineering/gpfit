from numpy import array, ones, hstack, zeros, tile, argmin, nonzero
from numpy.linalg import lstsq, matrix_rank
from numpy.random import permutation as randperm

def max_affine_init(x, y, K):
	'''
	initializes max-affine fit to data (y, x)
	ensures that initialization has at least K+1 points per partition (i.e.
	per affine function)
	'''
	
	defaults = {}
	defaults['bverbose'] = True
	options = defaults

	npt,dimx = x.shape
	X = hstack((ones((npt, 1)), x))
	ba = zeros((dimx+1,K))

	if K*(dimx+1) > npt:
		raise Exception('Not enough data points')

	# Choose K unique indices
	randinds = randperm(npt)[0:K]

	# partition based on distances
	sqdists = zeros((npt, K))
	for k in range(K):
		sqdists[:,k] = ((x - tile(x[randinds[k],:],(npt, 1))) ** 2).sum(1) 

	#index to closest k for each data pt 
	mindistind = argmin(sqdists,axis=1)

	'''
	loop through each partition, making local fits
	note we expand partitions that result in singular least squares problems
	why this way? some points will be shared by multiple partitions, but
	resulting max-affine fit will tend to be good. (as opposed to solving least-norm version)
	'''
	for k in range(K): #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ask woody about this

	    inds = mindistind == k

	    #before fitting, check rank and increase partition size if necessary
	    #(this does create overlaps)
	    if matrix_rank(X[inds, :]) < dimx + 1:
	        sortdistind = sqdists[:,k].argsort()

	        i = sum(inds)  #i is number of points in partition
	        iinit = i

	        if i < dimx+1:
	            #obviously, at least need dimx+1 points. fill these in before
	            #checking any ranks
	            inds[sortdistind[i+1:dimx+1]] = 1 #<<<<<<<<<<<check index
	            i = dimx+1 #<<<<<<<<<<<<<<<<<<<check index

	        #now add points until rank condition satisfied
	        while matrix_rank(X[inds, :]) < dimx+1:
	            i = i+1
	            inds[sortdistind[i]] = 1
	        
	        if options['bverbose']:
	        	print('max_affine_init: Added ' + repr(i-iinit) + ' points to partition ' + repr(k) + ' to maintain full rank for local fitting.')
	    
	    #now create the local fit
	    ba[:,k] = lstsq(X[inds.nonzero()], y[inds.nonzero()])[0][:,0]

	return ba
