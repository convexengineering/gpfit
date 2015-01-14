def process_options(defaults, varargin):
	options = defaults
	
	#process inputs
	i = 0 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< changed to 0 from 1
	while(i < len(varargin)):
	    options[varargin[i]] = varargin[i+1] #<<<<<<<<<<<<<<<<<<<<  curly braces
	    i = i+2 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  why +2

	return options
