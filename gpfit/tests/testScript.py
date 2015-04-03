from numpy import arange, array, vstack, hstack, newaxis, linspace, log, exp
from gpfit.compare_fits import compare_fits

whichtest = 'circuit'
#1D ###########################################
if whichtest == 'eg1':
    x = arange(0.,16.)[:,newaxis]
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

#2D ###########################################
if whichtest =='eg2':
    x1 = arange(0.,16.)[:,newaxis]
    x2 = x1**3
    x = hstack((x1,x2))
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

#3D ###########################################
if whichtest =='eg3':
    x1 = arange(0.,16.)[:,newaxis]
    x2 = x1**2
    x3 = x1**3
    x = hstack((x1,x2,x3))
    y1 = arange(0.,11.)[:,newaxis]
    y2 = arange(12.,21.,2.)[:,newaxis]
    y = vstack((y1,y2))
    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

#Circuit #####################################
if whichtest == 'circuit':
    vdd = linspace(1.0,2.0,10)
    vth = linspace(0.2,0.4,10)
    VDD, VTH = meshgrid(vdd, vth)
    VDD = VDD.reshape(10,1)
    VTH = VTH.reshape(10,1)
    u = hstack(VDD,VTH)
    x = log(u)

    P = VDD**2 + 30*VDD*exp(-(VTH-0.06*VDD)/0.039)
    w = P

    x = log(u)
    y = log(w)

    Ks = array([2])
    ntry = 1
    s = compare_fits(x,y,Ks,ntry)

print s['max_affine']['params']
print s['softmax_optMAinit']['params']
print s['softmax_originit']['params']
print s['implicit_originit']['params']

########################################################################################################################################
