# Plotting:
# Importing necessary packages for plotting
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import re
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

def retrive_data_from_file(filename):
    infile = open(filename, 'r')

    lambda_values = []
    eta_values = []
    MSE_average = []
    R2_average = []
    Bias_average = []
    Variance_average =[]

    for lines in infile: 
        line = lines.replace(':',' ').replace('[', ' ').replace(',', ' ').replace(']', ' ')
        l=line.split()
        if l[0] == 'The_results_from_running_with_lamda':
            lambda_values =np.float_(l[1::])
        if l[0] == 'Etas':
            eta_values =np.float_(l[1::])
        if l[0] == 'MSE_average':
            MSE_average =np.float_(l[1::])
        if l[0] == 'R2_score_average':
            R2_average =np.float_(l[1::])
        if l[0] == 'Bias_average':
            Bias_average =np.float_(l[1::])
        if l[0] == 'Variance_average':
            Variance_average =np.float_(l[1::])

    return MSE_average, R2_average, Bias_average, Variance_average, lambda_values, eta_values


def plot_MSE_Bias_Var(mse, bias, var,lambda_values, eta_values, m):
    if m=='Ridge' or m=='Lasso' or m=='OLS':
        plt.plot(lambda_values, mse)
        plt.plot(lambda_values, bias)
        plt.plot(lambda_values, var)
        plt.xlabel(r'$\lambda$', fontsize=18)

    elif m== 'NN':
        plt.plot(eta_values, mse)
        plt.plot(eta_values, bias)
        plt.plot(eta_values, var)
        plt.xlabel(r'$\eta$', fontsize=18)

    plt.title('Metrics calculated by %s' %m, fontsize=20)
    pylab.xticks(fontsize=14)
    pylab.yticks(fontsize=14)
    plt.semilogx()
    plt.legend(['MSE', r'Bias$^2$', 'Variance'], fontsize=18)
    plt.show()

def plot_R2score(R2_OLS, R2_Ridge, R2_Lasso, lambda_values):
    plt.plot(lambda_values, R2_OLS, 'o')
    plt.plot(lambda_values, R2_Ridge)
    plt.plot(lambda_values, R2_Lasso)
    plt.legend(['OLS', 'Ridge', 'Lasso'], fontsize=18)
    plt.title(r'R$^2$-score calculated by different methods', fontsize=20)
    plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel(r'R$^2$-score',fontsize=18)
    pylab.xticks(fontsize=12)
    pylab.yticks(fontsize=12)
    plt.semilogx()
    plt.show()

def plot_R2score_NN(R2, eta_values):
    plt.plot(eta_values, R2)
    plt.title(r'R$^2$-score calculated by neural network', fontsize=20)
    plt.xlabel(r'$\eta$', fontsize=18)
    plt.ylabel(r'R$^2$-score',fontsize=18)
    pylab.xticks(fontsize=12)
    pylab.yticks(fontsize=12)
    plt.semilogx()
    plt.show()

def plot_Jstates(J, method, lambda_value, L):
    J_leastsq=np.array(J).reshape((L,L))
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    im = axarr.imshow(J_leastsq,**cmap_args)
    if method == 'J-states':
        axarr.set_title('J-states',fontsize=16)
    else:
        axarr.set_title(r'%s $\lambda =$ %s'%(method, lambda_value),fontsize=16)
    axarr.tick_params(labelsize=16)
    divider = make_axes_locatable(axarr)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar=fig.colorbar(im, cax=cax)

    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
    cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
    plt.show()

def Plot_Accuracy(acc, eta):

    xaxis = np.linspace(0,len(acc)-1, len(acc))
    plt.plot(xaxis , acc, 'bo', markersize=2, label='Training accuracy')
    plt.title(r"Accuracy for training on the 2D Ising-model with $\eta$ = %1.1e" % eta, fontsize=20)
    plt.xlabel("Number of iterations", fontsize=18)
    plt.ylabel("Percentage of correct predictions", fontsize=18)
    pylab.xticks(fontsize=14)
    pylab.yticks(fontsize=14)
    plt.legend(fontsize=18)
    plt.show()
