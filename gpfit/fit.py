from implicit_softmax_affine import implicit_softmax_affine
from LM import LM
from generic_resid_fun import generic_resid_fun
from max_affine_init import max_affine_init
from numpy import append, ones, size
from print_fit import print_ISMA, print_SMA, print_MA

def fit(xdata, ydata, K, ftype="ISMA", varNames=[]):

	# Check data is in correct form


	# Initialize fitting variables
	alphainit = 10
	bainit = max_affine_init(xdata, ydata, k)

	if ftype == "ISMA":

		def rfun (p): return generic_resid_fun(implicit_softmax_affine, xdata, ydata, p)
		[params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit*ones((k,1))))

		d = (params.size - K)/K - 1 #Number of independent dimensions (independent variables)

		alpha = 1./params[range(-K,0)]

		A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
		B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

		print_str = print_ISMA(A, B, alpha, d, K)

		# Create posynomial object

		# Plot fit if possible

		# Output data over specified range? something like:
		#w_ISMA, _ = implicit_softmax_affine(x,PAR_ISMA)
		#w_ISMA = exp(w_ISMA)

		# Output RMS error?

	elif ftype == "SMA":

		def rfun (p): return generic_resid_fun(softmax_affine, xdata, ydata, p)
		[params, RMStraj] = LM(rfun, append(bainit.flatten('F'), alphainit))

		d = (params.size - 1)/K - 1 #Number of independent dimensions (independent variables)

		alpha = 1./params[-1]

		A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
		B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

		print_str = print_SMA(A, B, alpha, d, K)


	elif ftype == "MA":

		d = params.size/K - 1 #Number of dimensions (independent variables)

		A = params[[i for i in range(K*(d+1)) if i % (d + 1) != 0]]
		B = params[[i for i in range(K*(d+1)) if i % (d + 1) == 0]]

		print_str = print_MA(A, B, d, K)


