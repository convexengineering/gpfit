from numpy import array, tile, exp, log

def lse_scaled(x,alpha):
	'''
	log sum exp function with derivatives
	sums across the second dimension of x; 
	returns one y for every row of x
	dydx gives the deriv of each y wrt each x 
	alpha is softness parameter \in R

	note that lse_scaled is a mapping R^n --> R, where n==size(x,2)
	'''
	
	_, n = x.shape

	m = x.max(axis=1)  #maximal x values

	h = x - (tile(m, (n, 1))).T # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #distance from m; note h <= 0 for all entries
	#(unless x has infs; in this case h has nans; can prob deal w/ this gracefully)
	#should also deal with alpha==inf case gracefully

	expo = exp(alpha*h)
	sumexpo = expo.sum(axis=1)

	L = log(sumexpo)/alpha

	y = L + m

	dydx = expo/(tile(sumexpo, (n, 1))).T #<<<<<<<<<<<<<<<<<<<<
	#note that sum(dydx,2)==1, i.e. dydx is a probability distribution
	dydalpha = ((h*expo).sum(axis=1)/sumexpo - L)/alpha
	
	return y, dydx, dydalpha