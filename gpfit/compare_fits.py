from max_affine_init import max_affine_init

def compare_fits(xdata, ydata, Ks, ntry):
# INPUTS
#	xdata:	list containing x data for fitting
#	ydata: 	list containing y data for fitting
#	Ks: 	list containing values of k????
#	ntry:	???

	#size(xdata) not included

	def store_results(fieldname):
		s[fieldname]['resid'][ik,t] = min(RMStraj)
		s[fieldname]['iter'][ik,t] = len(RMStraj) 
		s[fieldname]['params'][ik,t] = params #check {ik,t}
		s[fieldname]['time'][ik,t] = time.time() - t

	s = {}
	s['Ks'] = Ks;

	s[]

	alphainit = 10

	for ik in len(Ks):
		for t in len(ntry):
		
			k = Ks[ik]

			bainit = max_affine_init(xdata, ydata, k)
			bainit = sum(bainit,[]) # sum(bainit,[]) flattens bainit row-by-row

			def rfun (p): return generic_resid_fun(max_affine, xdata, ydata, p)
			t = time.time()
			[params, RMStraj] = LM(rfun, bainit) 
			store_results('max_affine')

			def rfun (p): return generic_resid_fun(softmax_affine, xdata, ydata, p)
			t = time.time()
			[params, RMStraj] = LM(rfun, [params(:); alphainit])
			store_results('softmax_optMAinit')

			def rfun (p): return generic_resid_fun(softmax_affine, xdata, ydata, p)
			t = time.time()
			[params, RMStraj] = LM(rfun, bainit.append(alphainit))
			store_results('softmax_originit')

			def rfun (p): return generic_resid_fun(implicit_softmax_affine, xdata, ydata, p)
			t = time.time()
			[params, RMStraj] = LM(rfun, bainit.append([alphainit]*k))		
			store_results('implicit_originit')

	return s