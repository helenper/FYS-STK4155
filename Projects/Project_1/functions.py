####################
# Project 1 - functions 
# FYS-STK 3155/4155
# Fall 2018 
####################

# Import necessary packages
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from time import time


def FrankeFunction(x,y):
    '''Returns the Franke function'''

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def polynomialfunction(x, y, n, degree):
    '''Returns the X-hat matrix for different degrees of 
    polynomials up to degree five'''

    if degree==1: 
        X = np.c_[np.ones((n,1)) , x, y]

    elif degree==2:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2]

    elif degree==3:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3]

    elif degree==4:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4]

    elif degree==5:
        X = np.c_[np.ones((n,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]
    else:
        print('Degree out of range!')

    return X

def OLS(X, z, X_test, z_test):
    '''Calculate and return the z and zpredict value by 
    ordinary least squares method'''

    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 
    zpredict = X_test.dot(beta) 
    return quality(z_test,zpredict)

def quality(z,zpredict, write=0):
    '''A function that calculate the mean square error and the R2 score of 
    the values sendt in. If the write value is anything else than zero
    the function will print out the values'''

    # Mean squared error:
    mse = mean_squared_error(z,zpredict)
    
    # Explained R2 score: 1 is perfect prediction      
    R2 = 1- (np.sum((z-zpredict)**2))/(np.sum((z-np.mean(z))**2))
    
    if write != 0:
        print('Mean square error: %.5f' % mse)
        print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict))
        print('R2 score: %.5f' % R2)
        print('R2 score scitkitlearn: %.5f' % r2_score(z, zpredict))

    return mse, R2


def ridge(X, z, X_test, z_test, alpha, write=0):
    ''' A function that implementes the Rigde method'''

    n_samples = 100

    IX = np.eye(X.shape[1])

    beta_ridge = (np.linalg.pinv( X.T @ X + alpha*IX) @ X.T @ z).flatten() 

    pred_ridge =  X_test @ beta_ridge # Shape: 100x6 from 6 lambda-values

    
    ### R2-score of the results
    if write:
        print('lambda = %g'%alpha)
        print('r2 for scikit: %g'%r2_score(z,pred_ridge_scikit[:,i]))
        print('r2 for own code, not centered: %g'%r2_score(z,pred_ridge))
        
    
    return quality(z_test, pred_ridge)


def lasso(X,z,X_test, z_test, alpha, write=0):
    ''' A function that implements the Lasso method'''

    lasso=Lasso(alpha)
    lasso.fit(X,z)
    predl=lasso.predict(X_test)

    if write != 0:
        print("Lasso Coefficient: ", lasso.coef_)
        print("Lasso Intercept: ", lasso.intercept_)
        print("R2 score:", r2_score(z,predl))

    return quality(z_test, predl)


def splitdata(data, percent):
    '''A function to implement the method of bootstrap resampeling method to 
    split data into train and test parts. The variable "percent" determins how many percents of
    the data is used to be traind on'''
    size = int(len(data)*percent)
    train = np.random.choice(len(data),size)
    test = list(set(range(len(data))) - set(train))
    return train, test

def bootstrap(x,y):

    indices = np.random.choice(len(x),len(x))
    x_train_new = x[indices]        
    y_train_new = y[indices]
    return x_train_new, y_train_new



