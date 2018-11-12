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

def retrive_data_from_file(filename, numb_of_lambda):
    infile = open(filename, 'r')

    MSE_average = []
    R2_average = []
    Bias_average = []
    Variance_average =[]

    i = 0

    for lines in infile: 
        line = lines.split()
        if len(line) == 14: # 12 for Ridge og Lasso!
            if line[0] == 'MSE_average:' :
                for i in range(12):
                    i += 2
                    if int(i)%2 == 0:
                        MSE_average.append(float(line[i]))
                    #print(MSE_average)
            
            if line[0] == 'R2_score_average:':
                for i in range(12):
                    i+=2
                    if int(i)%2 == 0:
                        R2_average.append(float(line[i]))

            if line[0] == 'Bias_average:' :
                for i in range(12):
                    i+=2
                    if int(i)%2 == 0:
                        Bias_average.append(float(line[i]))

            if line[0] == 'Variance_average:':
                for i in range(12):
                    i+=2
                    if int(i)%2 == 0:
                        Variance_average.append(float(line[i]))
        elif len(line) == 2:
            if line[0] == 'MSE_average:':
                for i in range(6):
                    MSE_average.append(float(line[1]))
            if line[0] == 'R2_score_average:':
                for i in range(6):
                    R2_average.append(float(line[1]))
            if line[0] == 'Bias_average:':
                for i in range(6):
                    Bias_average.append(float(line[1]))
            if line[0] == 'Variance_average:':
                for i in range(6):
                    Variance_average.append(float(line[1]))

        elif len(line) == 16:
            if line[0] == 'MSE_average:' :
                for i in range(14):
                    i += 2
                    if int(i)%2 == 0:
                        MSE_average.append(float(line[i]))
                    #print(MSE_average)
            
            if line[0] == 'R2_score_average:':
                for i in range(14):
                    i+=2
                    if int(i)%2 == 0:
                        R2_average.append(float(line[i]))

            if line[0] == 'Bias_average:' :
                for i in range(14):
                    i+=2
                    if int(i)%2 == 0:
                        Bias_average.append(float(line[i]))

            if line[0] == 'Variance_average:':
                for i in range(14):
                    i+=2
                    if int(i)%2 == 0:
                        Variance_average.append(float(line[i]))

            


    infile.close()

    return MSE_average, R2_average, Bias_average, Variance_average

def plot_MSE_Bias_Var(mse, bias, var, m):
    if m=='Ridge' or m=='Lasso' or m=='OLS':
        lambda_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        plt.plot(lambda_values, mse)
        plt.plot(lambda_values, bias)
        plt.plot(lambda_values, var)
        plt.xlabel(r'$\lambda$', fontsize=18)

    elif m== 'neural network':
        eta_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        plt.plot(eta_values, mse)
        plt.plot(eta_values, bias)
        plt.plot(eta_values, var)
        plt.xlabel(r'$\eta$', fontsize=18)

    plt.title('Metrics calculated by %s' %m, fontsize=20)
    plt.ylabel('Metrics', fontsize=18)
    pylab.xticks(fontsize=14)
    pylab.yticks(fontsize=14)
    plt.semilogx()
    plt.legend(['MSE', r'Bias$^2$', 'Variance'], fontsize=18)
    plt.show()


 #Getting the data from file:
mse_Ridge, r2_Ridge, bias_Ridge, var_Ridge = retrive_data_from_file('Ridge_seed12.txt', 5)
print(mse_Ridge)
mse_Lasso, r2_Lasso, bias_Lasso, var_Lasso = retrive_data_from_file('Lasso_seed12.txt', 5)
print(mse_Lasso, r2_Lasso)
mse_OLS, r2_OLS, bias_OLS, var_OLS = retrive_data_from_file('results_OneDim_OLS_1000iterations.txt', 5)
print('mse', mse_OLS)
mse_NN, r2_NN, bias_NN, var_NN = retrive_data_from_file('results_OneDim_NN.txt', 5)
print(mse_NN)

m = 'Ridge'
#plot_MSE_Bias_Var(mse_Ridge, bias_Ridge, var_Ridge, m)
m = 'Lasso'
#plot_MSE_Bias_Var(mse_Lasso, bias_Lasso, var_Lasso, m)
m = 'OLS'
#plot_MSE_Bias_Var(mse_OLS, bias_OLS, var_OLS, m)
m = 'neural network'
plot_MSE_Bias_Var(mse_NN, bias_NN, var_NN, m)




def plot_R2score(R2_OLS, R2_Ridge, R2_Lasso):
    lambda_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
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

#plot_R2score(r2_OLS, r2_Ridge, r2_Lasso)


def plot_R2score_NN(R2):
    eta_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    plt.plot(eta_values, R2)
    plt.title(r'R$^2$-score calculated by neural network', fontsize=20)
    plt.xlabel(r'$\eta$', fontsize=18)
    plt.ylabel(r'R$^2$-score',fontsize=18)
    pylab.xticks(fontsize=12)
    pylab.yticks(fontsize=12)
    plt.semilogx()
    plt.show()

plot_R2score_NN(r2_NN)

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
