import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_fit(u, w, w_fit):
    '''
    Plots fitted function and original data
    '''
    if u.shape[1] == 1:
        plt.plot(u,w,'+',u,w_fit,'-')
    elif u.shape[1] ==2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0],u[:,1],w)

    plt.show()


        # if plot == True:
        #     if d  == 1:
        #         u = exp(xdata)
        #         plot_fit(u, w, w_ISMA)
        #     elif d == 2:
        #         [X1 X2] = meshgrid(x[:,1],x[:,2])
        #         X1 = X1.reshape(X1.size,1)
        #         X2 = X2.reshape(X2.size,1)
        #         X = hstack((X1,X2))
        #         U = exp(X)
        #         y_fit, _ = implicit_softmax_affine(X, params)
        #         w_fit = exp(y_fit)
        #         plot_fit(U, w, w_fit)