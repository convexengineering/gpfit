from numpy import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MA
def plot_MA_fit(PAR_MA, K):


# SMA
def plot_SMA_fit(PAR_SMA, K):
	A = PAR_MA[[i for i in range(K) if i % 3 != 0]]
	B = PAR_MA[[i for i in range(K) if i % 3 == 0]]



# ISMA