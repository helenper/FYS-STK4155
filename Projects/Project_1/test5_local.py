####################
# Project 1 
# FYS-STK 3155/4155
# Fall 2018 
####################


####################
# Franke function - given in exercise
####################

# Import necessary packages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from time import time
import matplotlib.mlab as mlab


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

    

# Make the dataset
n = 100						# number of datapoints
x = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset with noise

def polynomialfunction(x, y, type):
    if type==1: 
        X = np.c_[np.ones((n,1)) , x, y]

    elif type==2:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2]

    elif type==3:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3]

    elif type==4:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4]

    elif type==5:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]
    else:
        print('Degree out of range!')

    return X

X = polynomialfunction(x,y,type=5)# Give your wish for degree as type

z = FrankeFunction(x, y) + 0.9*np.random.randn(1) # z with noise

def OLS(X, z):
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 
    zpredict = X.dot(beta) 
    return z, zpredict

z, zpredict = OLS(X,z)
# The mean squared error  

def quality(z,zpredict):

    mse = 1.0/z.shape[0] *np.sum((z - zpredict)**2)

    print('Mean square error easy: %.5f' % mse)
    print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict))

    # Explained variance score: 1 is perfect prediction      

    R2 = 1- (np.sum((z-zpredict)**2))/(np.sum((z-np.mean(z))**2))

    print('R2 score easy: %.5f' % R2)
    print('R2 score scitkitlearn: %.5f' % r2_score(z, zpredict))

    #Some other variances:
    var=1.0/z.shape[0] *np.sum((z - np.mean(z))**2)
    #betavar=1.0/z.shape[0] *np.sum((beta - np.mean(beta))**2)
    print('Variance: %.5f'% var)
    #print('Variance of beta', betavar)

# Ridge and Lasso:
def ridge(x, y, z, X):
    np.random.seed(4155)

    n_samples = 100

    x_ = x-np.mean(x)
    y_ = y-np.mean(y)
    z_ = z-np.mean(z) #Needed?

    X_ = np.c_[x, y, x**2, x*y, y**2, \

                    x**3, x**2*y, x*y**2, y**3, \

                    x**4, x**3*y, x**2*y**2, x*y**3,y**4, \

                    x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5] # Check this! What is this?

    lmb_values = [1e-4, 1e-3, 1e-2, 10, 1e2, 1e4]
    num_values = len(lmb_values)

    ## Ridge-regression of centered and not centered data
    beta_ridge = np.zeros((X.shape[1],num_values))
    beta_ridge_centered = np.zeros((X.shape[1],num_values))

    IX = np.eye(X.shape[1])
    IX_ = np.eye(X_.shape[1])

    for i,lmb in enumerate(lmb_values):
        beta_ridge[:,i] = (np.linalg.pinv( X.T @ X + lmb*IX) @ X.T @ z).flatten() #maybe change to pinv
        beta_ridge_centered[1:,i] = (np.linalg.pinv( X_.T @ X_ + lmb*IX_) @ X_.T @ z_).flatten() #pinv?

    beta_ridge_centered[0,:] = np.mean(z)

    pred_ridge =  X @ beta_ridge
    pred_ridge_centered =  X_ @ beta_ridge_centered[1:] + beta_ridge_centered[0,:]

    ### R2-score of the results
    for i in range(num_values):
        print('lambda = %g'%lmb_values[i])
        #print('r2 for scikit: %g'%r2_score(z,pred_ridge_scikit[:,i]))
        print('r2 for own code, not centered: %g'%r2_score(z,pred_ridge[:,i]))
        print('r2 for own, centered: %g\n'%r2_score(z,pred_ridge_centered[:,i]))


#Lasso:
def lasso(X,z, alpha):
    lasso=Lasso(alpha)
    lasso.fit(X,z)
    predl=lasso.predict(X)
    print("Lasso Coefficient: ", lasso.coef_)
    print("Lasso Intercept: ", lasso.intercept_)
    print("R2 score:", r2_score(z,predl))



"""
plt.scatter(X,z,color='green', label="Training Data")
plt.plot(X, predl, color='blue', label="Lasso")
plt.legend()
plt.show()
"""
"""
print(np.shape(x_train))
poly5 = PolynomialFeatures(5)
#X_train = poly5.fit_transform()

X_train = np.c_[x_train,y_train]
X_test = np.c_[x_test, y_test]
lasso=Lasso(alpha=0.1)
lasso.fit(X_train,z_train)
predl=lasso.predict(z_test)
print("Lasso Coefficient: ", lasso.coef_)
print("Lasso Intercept: ", lasso.intercept_)
"""


# Plotting:
# Importing necessary packages for plotting


def plotFrankeFunction(x, y, z, type):
    #x = np.sort(x)
    #y = np.sort(y)
    x, y = np.meshgrid(x, y)
    #z = z.reshape(-1,1)
    print(x.shape)
    print(z.shape)
    z = FrankeFunction(x,y)
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


def bootstrap(data, percent):
    size = percent*len(data)
    train = np.random.choice(len(data),int(size))
    test = list(set(range(len(data))) - set(train))
    return train, test

print(np.shape(X))

train_indices, test_indices = bootstrap(z, 0.7)
X_train = X[train_indices]#; print(X_train.shape)
X_test = X[test_indices]#; print(X_test.shape)
z_train = z[train_indices]#; print(z_train.shape)
z_test = z[test_indices]#; print(z_test.shape)









