# Plotting:
# Importing necessary packages for plotting
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure, show, plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter #, MaxNLocator
#import matplotlib.ticker as ticker

#def plotStatistics()

def surface_plot(surface,title1, title2, surface1=None):
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title1)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.viridis,linewidth=0)
        plt.title(title2)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title1)


def retrive_data_from_file(filename, num_degree, numb_of_lambda):
    infile = open(filename, 'r')

    #lamda_values = np.zeros((num_degree, numb_of_lambda))
    #MSE_average = np.zeros((num_degree,numb_of_lambda))
    MSE_average = []
    R2_average = []
    Bias_average = []
    Variance_average =[]


    #i = 0

    for lines in infile: 
        line = lines.split()
        """
        if len(line) > 10:
            if line
            print('a') 
            lamda_values[i] = line[11]
        """
        #for i in range(numb_of_lambda):
        if len(line) == 2:
            #print(line[1])
            if line[0] == 'MSE_average:' :
                MSE_average.append(float(line[1]))
            
            if line[0] == 'R2_score_average:':
                R2_average.append(line[1])

            if line[0] == 'Bias_average:' :
                Bias_average.append(line[1])

            if line[0] == 'Variance_average:':
                Variance_average.append(line[1])
            


    infile.close()

    return MSE_average, R2_average, Bias_average, Variance_average

#MSE_average, R2_average, Bias_average, Variance_average = retrive_data_from_file('results_franke_OLS.txt', 5, 6)
#print(MSE_average, R2_average, Bias_average, Variance_average)
"""
def plotMSE_OLS(mse):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    print(mse[0:6])
    print(mse[6:12])
    plt.plot(lambda_values, mse[0:6], 'o')
    plt.plot(lambda_values, mse[6:12], 'o')
    plt.plot(lambda_values, mse[12:18], 'o')
    plt.plot(lambda_values, mse[18:24], 'o')
    plt.plot(lambda_values, mse[24:30], 'o')
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title('MSE calculated by OLS')
    plt.xlabel(r'$\lambda$')
    plt.semilogx()
    plt.ylabel('MSE')
    #plt.show()
"""
#mse_OLS, r2_OLS, bias_OLS, var_OLS = retrive_data_from_file('results_franke_OLS.txt', 5, 6)
#plotMSE_OLS(mse)


def plotMSE_Ridge(mse, mse_OLS):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse[0:6], 'b')
    plt.plot(lambda_values, mse[6:12], 'c')
    plt.plot(lambda_values, mse[12:18], 'g')
    plt.plot(lambda_values, mse[18:24], 'r')
    plt.plot(lambda_values, mse[24:30], 'm')
    plt.plot( lambda_values, mse_OLS[0:6], 'bo')
    plt.plot(lambda_values, mse_OLS[6:12], 'co')
    plt.plot(lambda_values, mse_OLS[12:18], 'go')
    plt.plot(lambda_values, mse_OLS[18:24], 'ro')
    plt.plot(lambda_values, mse_OLS[24:30], 'mo')

    plt.title('MSE calculated by Ridge')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.semilogx()
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.show()

#mse, r2, bias, var = retrive_data_from_file('results_franke_Ridge.txt', 5, 6)
#plotMSE_Ridge(mse, mse_OLS)

def plotMSE_Lasso(mse):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse[0:6])
    plt.plot(lambda_values, mse[6:12])
    plt.plot(lambda_values, mse[12:18])
    plt.plot(lambda_values, mse[18:24])
    plt.plot(lambda_values, mse[24:30])
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title('MSE calculated by Lasso')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.show()

#mse, r2, bias, var = retrive_data_from_file('results_franke_Lasso_n10.txt', 5, 6)
#plotMSE_Lasso(mse)

def plotR2_OLS():
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse_1)
    plt.plot(lambda_values, mse_2)
    plt.plot(lambda_values, mse_3)
    plt.plot(lambda_values, mse_4)
    plt.plot(lambda_values, mse_5)
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title(r'R$^2$ score calculated by OLS')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'R$^2$ score')
    plt.show()

def plotR2_Ridge():
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse_1)
    plt.plot(lambda_values, mse_2)
    plt.plot(lambda_values, mse_3)
    plt.plot(lambda_values, mse_4)
    plt.plot(lambda_values, mse_5)
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title(r'R$^2$ score calculated by Ridge')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'R$^2$ score')
    plt.show()

def plotR2_Lasso():
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse_1)
    plt.plot(lambda_values, mse_2)
    plt.plot(lambda_values, mse_3)
    plt.plot(lambda_values, mse_4)
    plt.plot(lambda_values, mse_5)
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title(r'R$^2$ score calculated by Lasso')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'R$^2$ score')
    plt.show()    

def plotOLS():#, mse_Ridge, mse_Lasso):
    poly = [i+1 for i in range(len(mse))]
    plt.plot(poly,mse)
    plt.plot(poly, r2)
    plt.plot(poly, bias)
    plt.plot(poly, var)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with OLS')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()

def plotRidge():#, mse_Ridge, mse_Lasso):
    poly = [i+1 for i in range(len(mse))]
    plt.plot(poly,mse)
    plt.plot(poly, r2)
    plt.plot(poly, bias)
    plt.plot(poly, var)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with Ridge')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()

def plotLasso():#, mse_Ridge, mse_Lasso):
    poly = [i+1 for i in range(len(mse))]
    plt.plot(poly,mse)
    plt.plot(poly, r2)
    plt.plot(poly, bias)
    plt.plot(poly, var)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with Lasso')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()

"""
def plotBias(bias_OLS,bias_Ridge,bias_Lasso):
    poly = [i+1 for i in range(len(mse_OLS))]
    plt.plot(poly,bias_OLS)
    plt.plot(poly, bias_Ridge)
    plt.plot(poly,bias_Lasso)
    plt.legend(['OLS', 'Ridge', 'Lasso'])
    plt.title('Bias calculated with different methods')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Bias')
    plt.show()  

def plotVariance(var_OLS,var_Ridge,var_Lasso):
    poly = [i+1 for i in range(len(mse_OLS))]
    plt.plot(poly,var_OLS)
    plt.plot(poly, var_Ridge)
    plt.plot(poly,var_Lasso)
    plt.legend(['OLS', 'Ridge', 'Lasso'])
    plt.title('Variance calculated with different methods')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Variance')
    plt.show()

def plotR2(r2_OLS, r2_Ridge, r2_Lasso):
    poly = [i+1 for i in range(len(r2_OLS))]
    plt.plot(poly, r2_OLS)
    plt.plot(poly, r2_Ridge)
    plt.plot(poly, r2_Lasso)
    plt.legend(['OLS','Ridge','Lasso'])
    plt.title('R2 score calculated with different methods')
    plt.xlabel('Polynomial degree')
    plt.ylabel('R2 score')
    plt.show()


def plotFrankeFunction(x, y, z, type):
    #x = np.sort(x)
    #y = np.sort(y)
    #x, y = np.meshgrid(x, y)
    #x = x.reshape(-1,1)
    #y = y.reshape(-1,1)
    #z = z.reshape(-1,1)
    print(x.shape)
    print(z.shape)
    #z = FrankeFunction(x,y)
    #n,m = x.shape 
    #z = z.reshape(n,m)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if type==1:
        plt.title('Franke function with actual z')
    elif type==2:
        plt.title('Franke function with our prediction of z')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
    return surf

# Plotting Franke function with actual z:
#plotFrankeFunction(x, y, z, type=1)

# Plotting Franke function with our prediction of z:
#plotFrankeFunction(x, y, zpredict, type=2)
"""
"""
# Plotting Ridge:
def plotRidge(x,y):
    # Sorting
    sort_ind = np.argsort(x[:,0])

    x_plot = x[sort_ind,0]
    x_centered_plot = x_[sort_ind,0]

    pred_ls_plot = pred_ls[sort_ind,0]
    pred_ridge_plot = pred_ridge[sort_ind,:]
    pred_ridge_centered_plot = pred_ridge_centered[sort_ind,:]

    # Plott not centered
    plt.plot(x_plot,pred_ls_plot,label='ls')

    for i in range(num_values):
        plt.plot(x_plot,pred_ridge_plot[:,i],label='ridge, lmb=%g'%lmb_values[i])

    plt.plot(x,y,'ro')

    plt.title('linear regression on un-centered data')
    plt.legend()

    # Plott centered
    plt.figure()

    for i in range(num_values):
        plt.plot(x_centered_plot,pred_ridge_centered_plot[:,i],label='ridge, lmb=%g'%lmb_values[i])

    plt.plot(x_,y,'ro')

    plt.title('linear regression on centered data')
    plt.legend()


    # 2.

    pred_ridge_scikit =  np.zeros((n_samples,num_values))
    for i,lmb in enumerate(lmb_values):
        pred_ridge_scikit[:,i] = (Ridge(alpha=lmb,fit_intercept=False).fit(X,y).predict(X)).flatten() # fit_intercept=False fordi bias er allerede i X

    plt.figure()

    plt.plot(x_plot,pred_ls_plot,label='ls')

    for i in range(num_values):
        plt.plot(x_plot,pred_ridge_scikit[sort_ind,i],label='scikit-ridge, lmb=%g'%lmb_values[i])

    plt.plot(x,y,'ro')
    plt.legend()
    plt.title('linear regression using scikit')

    plt.show()
    return 

plotRidge(x, y)
"""

"""
plt.scatter(X,z,color='green', label="Training Data")
plt.plot(X, predl, color='blue', label="Lasso")
plt.legend()
plt.show()
"""
