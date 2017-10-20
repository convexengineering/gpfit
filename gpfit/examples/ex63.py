import unittest
from numpy import log, exp, log10, vstack
from gpfit.fit import fit
from numpy.random import random_sample

Vdd = random_sample(1000,) + 1
Vth = 0.2*random_sample(1000,) + 0.2
P = Vdd**2 + 30*Vdd*exp(-(Vth-0.06*Vdd)/0.039)
u = vstack((Vdd,Vth))
x = log(u)
y = log(P)
K = 4

cstrt, rmsErr = fit(x,y,K,"ISMA")