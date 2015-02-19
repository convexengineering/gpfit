from numpy import arange, array, vstack, hstack, newaxis, linspace, log, exp, meshgrid
from gpfit.compare_fits import compare_fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


vdd = linspace(1.0,2.0,10)
vth = linspace(0.2,0.4,10)
VDD, VTH = meshgrid(vdd, vth)

P = VDD**2 + 30*VDD*exp(-(VTH-0.06*VDD)/0.039)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(VDD, VTH, P)
plt.show()

VDD = VDD.reshape(10,1)
VTH = VTH.reshape(10,1)
u = hstack(VDD,VTH)
x = log(u)


w = P.reshape(10,1)

x = log(u)
y = log(w)

Ks = array([2])
ntry = 1
s = compare_fits(x,y,Ks,ntry)
