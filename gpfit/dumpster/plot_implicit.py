from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

def plot_implicit(fn, (xmin,xmax),(ymin,ymax),(zmin,zmax)):
# def plot_implicit(fn, bbox = (-20,20)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    
    Ref: http://stackoverflow.com/questions/4680525/plotting-implicit-equations-in-3d
    '''

    # xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = np.linspace(xmin, xmax, 100) # resolution of the contour
    yy = np.linspace(ymin, ymax, 100)
    zz = np.linspace(zmin, zmax, 100)
    
    a1,a2 = np.meshgrid(xx,yy) # grid on which the contour is plotted
    b = np.linspace(xmin, xmax, 50) # number of slices
    c1,c2 = np.meshgrid(xx,zz) # grid on which the contour is plotted
    d = np.linspace(ymin, ymax, 50) # number of slices
    e1,e2 = np.meshgrid(yy,zz) # grid on which the contour is plotted
    f = np.linspace(zmin, zmax, 50) # number of slices

    for z in f: # plot contours in the XY plane
        X,Y = a1,a2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in d: # plot contours in the XZ plane
        X,Z = c1,c2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in b: # plot contours in the YZ plane
        Y,Z = e1,e2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    plt.show()