from numpy import arange, array, vstack, newaxis
from matplotlib.pyplot import figure, plot, draw
from gpfit import compare_fits


x = arange(0.,16.)[:,newaxis]

y1 = arange(0.,11.)[:,newaxis]
y2 = arange(12.,21.,2.)[:,newaxis]

y = vstack((y1,y2))


Ks = array([2])

ntry = array([1.])

s = compare_fits(x,y,Ks,ntry)

print s['max_affine']['params']
print s['softmax_optMAinit']['params']
print s['softmax_originit']['params']
print s['implicit_originit']['params']

########################################################################################################################################