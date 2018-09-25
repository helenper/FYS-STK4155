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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


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

num_rows = n 	# Choose this to be number of x points
num_cols = n 	# Choose this to be number of y points

X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \

                x**3, x**2*y, x*y**2, y**3, \

                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \

                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]



z = FrankeFunction(x, y) + 0.9*np.random.randn(1) # z with noise
print(np.shape(z))

beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 
print(np.shape(beta))


"""
zpredict = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        x_value = x[i]
        y_value = y[j]
        zpredict[j,i] = np.array([1 , x_value, y_value, x_value**2, x_value*y_value, y_value**2, \

                x_value**3, x_value**2*y_value, x_value*y_value**2, y_value**3, \

                x_value**4, x_value**3*y_value, x_value**2*y_value**2, x_value*y_value**3,y_value**4, \

                x_value**5, x_value**4*y_value, x_value**3*y_value**2, x_value**2*y_value**3,x_value*y_value**4, y_value**5]).dot(beta)
print(np.shape(zpredict))
"""
zpredict = X.dot(beta) #Is this correct? or should we do as above?

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

print('Variance score easy: %.5f' % R2)
print('Variance score: %.5f' % R_2(z, zpredict))
print('Variance score scitkitlearn: %.5f' % r2_score(z, zpredict))
"""
#Some other variances:
var=1.0/z.shape[0] *np.sum((z - np.mean(z))**2)
betavar=1.0/z.shape[0] *np.sum((beta - np.mean(beta))**2)
print('Variance:', var)
print('Variance of beta', betavar)
"""

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
    beta_ridge[:,i] = (np.linalg.inv( X.T @ X + lmb*IX) @ X.T @ z).flatten() #maybe change to pinv
    beta_ridge_centered[1:,i] = (np.linalg.inv( X_.T @ X_ + lmb*IX_) @ X_.T @ z_).flatten() #pinv?

# sett beta_0 = np.mean(z)
beta_ridge_centered[0,:] = np.mean(z)

## OLS (ordinary least squares) solution 
beta_ls = np.linalg.inv( X.T @ X ) @ X.T @ z #pinv?

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

"""
#Lasso:


lasso=linear_model.Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
predl=lasso.predict(X_test)
print("Lasso Coefficient: ", lasso.coef_)
print("Lasso Intercept: ", lasso.intercept_)

"""





