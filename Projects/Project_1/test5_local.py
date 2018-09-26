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
n = 10							# number of datapoints
x = np.random.uniform(0.0,1.0, n*n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n*n)       # create a random number for y-values in dataset with noise

#x = np.sort(x)
#y = np.sort(y)

num_rows = n 	# Choose this to be number of x points
num_cols = n 	# Choose this to be number of y points

X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \

                x**3, x**2*y, x*y**2, y**3, \

                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \

                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]



z = FrankeFunction(x, y) + 0.9*np.random.randn(1) # z with noise

beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 

zpredict = X.dot(beta) 

# The mean squared error  

def MSE(y, y_tilde):
    mse_calc = 0
    for i in range(len(y)):
        mse_calc += (y[i] - y_tilde[i])**2
    return mse_calc/len(y)

mse = 1.0/z.shape[0] *np.sum((z - zpredict)**2)

print('Mean square error easy: %.5f' % mse)
print('Mean squared error: %.5f' % MSE(z, zpredict))
print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict))

# Explained variance score: 1 is perfect prediction      
def R_2(y, y_tilde):
    y_mean = np.mean(y)
    r2_calc_up = 0
    r2_calc_down = 0
    for i in range(len(y)):
        r2_calc_up += (y[i] - y_tilde[i])**2
        r2_calc_down += (y[i]-y_mean)**2
    r2_calc = r2_calc_up/r2_calc_down
    return 1-r2_calc

R2 = 1- (np.sum((z-zpredict)**2))/(np.sum((z-np.mean(z))**2))

print('R2 score easy: %.5f' % R2)
print('R2 score: %.5f' % R_2(z, zpredict))
print('R2 score scitkitlearn: %.5f' % r2_score(z, zpredict))

#Some other variances:
var=1.0/z.shape[0] *np.sum((z - np.mean(z))**2)
#betavar=1.0/z.shape[0] *np.sum((beta - np.mean(beta))**2)
print('Variance: %.5f'% var)
#print('Variance of beta', betavar)


# Ridge and Lasso:
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

# sett beta_0 = np.mean(z)
beta_ridge_centered[0,:] = np.mean(z)

## OLS (ordinary least squares) solution 
beta_ls = np.linalg.pinv( X.T @ X ) @ X.T @ z #pinv?

## Evaluate the models
pred_ls = X @ beta_ls
pred_ridge =  X @ beta_ridge
pred_ridge_centered =  X_ @ beta_ridge_centered[1:] + beta_ridge_centered[0,:]

### R2-score of the results
for i in range(num_values):
    print('lambda = %g'%lmb_values[i])
    #print('r2 for scikit: %g'%r2_score(z,pred_ridge_scikit[:,i]))
    print('r2 for own code, not centered: %g'%r2_score(z,pred_ridge[:,i]))
    print('r2 for own, centered: %g\n'%r2_score(z,pred_ridge_centered[:,i]))


#Lasso:


lasso=Lasso(alpha=0.001)
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

"""
x_train=x[:37, np.newaxis]
x_test=x[37:, np.newaxis]
y_train=y[:37, np.newaxis]
y_test=y[37:, np.newaxis]
z_train=z[:37, np.newaxis]
z_test=z[37:, np.newaxis]

def stat(data):
    return np.mean(data)

def Bootstrap(data, statistics, N):
    t = np.zeros(N)
    inds = np.arange(len(data))
    t0 = time()

    for i in range(N):
        t[i] = statistics(data[np.random.randint(0, len(data), len(data))])

    print("Runtime: %g sec" % (time()-t0)); print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (statistics(data), np.std(data), np.mean(t), np.std(t)))

    return t

mu, sigma = 100, 15
datapoints = 10000

t = Bootstrap(X, stat, datapoints)

n, binsboot, patches = plt.hist(t,50,normed=1,facecolor='green', alpha=0.75)

y = mlab.normpdf(binsboot, np.mean(t), np.std(t))
#lt = plt.plot(binsboot, z, 'g--', linewidth=1)

plt.show()
"""



def bootstrap(data, percent):
    size = percent*len(data)
    train = np.random.choice(len(data),int(size))
    #test = list(set(range(len(data)) - set(train)))
    #test = [i for i in range(len(data)) if i!= train[i]]
    return train#, test

print(bootstrap(X, 0.7).shape[0])
print(bootstrap(X, 0.7))
print(np.shape(X))


for train_index in bootstrap(X, 0.7):
    X_train= X[train_index] # we want the whole fucking row! Not only the 

print(np.shape(X_train))
print(X_train)
"""
for train_index, test_index in bootstrap(X,0.7):
    X_train[train_index] = X[train_index]
    X_test[test_index] = X[test_index]
"""








