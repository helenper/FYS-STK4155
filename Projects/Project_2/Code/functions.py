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
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import safe_indexing, indexable
#from plotfunctions import *
import re


def OLS(X, z, X_test, z_test):
    '''Calculate and return the z and zpredict value by 
    ordinary least squares method'''
    beta = (np.linalg.pinv(X.T @ X)@ X.T @ z)
    zpredict = X_test @ beta
    mse, R2, bias, variance = quality(z_test, zpredict) 
    return mse, R2, bias, variance, beta

def quality(z_test,zpredict):
    '''A function that calculate the mean square error and the R2 score of 
    the values sendt in. If the write value is anything else than zero
    the function will print out the values'''

    # Mean squared error:
    mse = (1.0/(np.size(z_test))) *np.sum((z_test - zpredict)**2)
    # Explained R2 score: 1 is perfect prediction 
    R2 = 1- ((np.sum((z_test-zpredict)**2))/(np.sum((z_test-np.mean(z_test))**2)))
    # Bias:
    bias = np.mean((z_test - np.mean(zpredict, keepdims=True))**2)
    # Variance:
    variance = np.mean(np.var(zpredict, keepdims=True))
    
    return mse, R2, bias, variance


def ridge(X, z, X_test, z_test, lambda_value):
    ''' A function that implementes the Rigde method'''
    IX = np.eye(X.shape[1])
    beta_ridge = (np.linalg.pinv( X.T @ X + lambda_value*IX) @ X.T @ z) 
    pred_ridge =  X_test @ beta_ridge 
    mse, R2, bias, variance = quality(z_test, pred_ridge)
    return mse, R2, bias, variance, beta_ridge


def lasso(X,z,X_test, z_test, lambda_value):
    ''' A function that implements the Lasso method'''

    lasso=Lasso(lambda_value, max_iter=1e7, normalize = True, fit_intercept = False)
    lasso.fit(X,z) 
    beta_lasso = lasso.coef_.T
    predl=lasso.predict(X_test)

    mse, R2, bias, variance = quality(z_test, predl)
    return mse, R2, bias, variance, beta_lasso


def bootstrap(x,y):

    indices = np.random.choice(len(y),len(y))
    x_train_new = x[indices]        
    y_train_new = y[indices]
    return x_train_new, y_train_new

 