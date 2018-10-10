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


def splitdata(data, percent):
    '''A function to implement the method of bootstrap resampeling method to 
    split data into train and test parts. The variable "percent" determins how many percents of
    the data is used to be traind on'''
    size = int(len(data)*percent)
    train = np.random.choice(len(data),size)
    test = list(set(range(len(data))) - set(train))
    return train, test

def bootstrap(x,y):

    indices = np.random.choice(len(y),len(y))
    x_train_new = x[indices]        
    y_train_new = y[indices]
    return x_train_new, y_train_new

def betaConfidenceInterval(beta, best_beta, iteration_best):
    confidenceInterval = []
    for i in range(best_beta.shape[0]):
        sigma = np.sqrt(np.var(beta[iteration_best][i]))
        confidenceInterval_start = np.mean(best_beta[i]) - 2*sigma
        confidenceInterval_end = np.mean(best_beta[i]) + 2*sigma
        confidenceInterval.append([confidenceInterval_start, confidenceInterval_end])
    
    return confidenceInterval

def betaConfidenceInterval_terrain(beta, best_beta, iteration_best):
    confidenceInterval = []
    for i in range(len(best_beta)):
        sigma = np.sqrt(np.var(beta[i]))
        confidenceInterval_start = np.mean(best_beta[i]) - 2*sigma
        confidenceInterval_end = np.mean(best_beta[i]) + 2*sigma
        confidenceInterval.append([confidenceInterval_start, confidenceInterval_end])
    
    return confidenceInterval  

def runFranke(polydegree, lambda_values, num_data, num_iterations,seed, method):
    if seed == 'True' or seed == 'true':
        np.random.seed(4155)
        print('NOTE: You are running with a given seed on random data.')
    else:
        print('NOTE: You are running with random data.')

    n = num_data                              # number of datapoints
    row = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
    col = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset
    noise = 0.1                               # strengt of noise

    [C,R] = np.meshgrid(col, row)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    z = FrankeFunction(x,y) + noise*np.random.randn(len(x))

    #---------------------------------------------------------------------
    # Use bootstrap to define train and test data and calculate a mean 
    # value for MSE and R2 for the different methods OSL, Ridge and Lasso
    #---------------------------------------------------------------------

    iterations = num_iterations    # number of times we split and save our calculations in train and test point

    X = polynomialfunction(x,y,len(x),degree=polydegree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)

    # Create arrays to hold different values to be taken mean over later. 
    # Each arrray is a nested array, where the first index points to the degree of the polynomial
    # used in that iteration. 
    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta_list = [] 

    beta= 0
    best_beta = 0

    mse_min = 1000
    r2_for_min_mse = 0

    k = 0

    for i in range(iterations):
        X_train, z_train = bootstrap(X_train,z_train)
        if method == 'OLS':
            print(i)
            mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
            beta_list.append(beta)

        if method == 'Ridge':
            mse[i], r2score[i], bias[i], var[i], beta = ridge(X_train,z_train,X_test,z_test,lambda_values)
            beta_list.append(beta)
        
        if method == 'Lasso':
            mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,z_train,X_test,z_test,lambda_values)
            beta_list.append(beta)

        if mse[i] < mse_min: 
            mse_min = mse[i]
            r2_for_min_mse = r2score[i]
            best_beta = beta
            iteration_best = i

    mse_average = np.mean(mse[k])
    r2score_average = np.mean(r2score[k])
    bias_average = np.mean(bias[k])    
    var_average = np.mean(var[k])  

    return mse_average, r2score_average, bias_average, var_average, np.array(beta_list), best_beta, mse_min, r2_for_min_mse, iteration_best

def predict(rows, cols, beta):
    ''' Made by Kristine '''
    out = np.zeros((np.size(rows), np.size(cols)))

    for i,y_ in enumerate(rows):
        for j,x_ in enumerate(cols):
            data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2, \
                                x_**3, x_**2*y_, x_*y_**2, y_**3, \
                                x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4, \
                                x_**5, x_**4*y_, x_**3*y_**2, x_**2*y_**3,x_*y_**4,y_**5])
            out[i,j] = data_vec @ beta

    return out

def runTerrain(polydegree, lambda_values, num_data, num_iterations,seed, method):
    if seed == 'True' or seed == 'true':
        np.random.seed(4155)
        print('NOTE: You are running with a given seed on random data.')
    else:
        print('NOTE: You are running with random data.')

    terrain1 = imread('/Users/monaanderssen/Documents/Master/1.semester/FYS-STK4155/FYS-STK4155-git/Projects/Project_1/TerrainData/terrainone.tif')
    [n,m] = terrain1.shape

    patch_size_row = 100
    patch_size_col = 50

    rows = np.linspace(0,1,patch_size_row)
    cols = np.linspace(0,1,patch_size_col)

    [C,R] = np.meshgrid(cols,rows)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    num_data = patch_size_row*patch_size_col
    num_patches = 5

    row_starts = np.random.randint(0,n-patch_size_row,num_patches)
    col_starts = np.random.randint(0,m-patch_size_col,num_patches)

    iterations = num_iterations

    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta_list = [] 

    mse_average = np.zeros(num_patches)
    r2score_average = np.zeros(num_patches)
    bias_average = np.zeros(num_patches)
    var_average = np.zeros(num_patches)

    beta= 0
    best_beta = [[], [], [], [], []]

    mse_min = [1e7, 1e7, 1e7, 1e7, 1e7]
    r2_for_min_mse = [0, 0, 0, 0, 0]

    for k,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
        row_end = row_start + patch_size_row
        col_end = col_start + patch_size_col
        print(k)
        patch = terrain1[row_start:row_end, col_start:col_end]

        z = patch.reshape(-1,1)
        X = polynomialfunction(x,y,len(x),degree=polydegree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)
        for i in range(iterations):
            X_train, z_train = bootstrap(X_train, z_train)
            if method == 'OLS':
                mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
                beta_list.append(beta)

            if method == 'Ridge':
                mse[i], r2score[i], bias[i], var[i], beta = ridge(X_train,z_train,X_test,z_test,lambda_values)
                beta_list.append(beta)
            
            if method == 'Lasso':
                mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,z_train,X_test,z_test,lambda_values)
                beta_list.append(beta)

            if mse[i] < mse_min[k]: 
                mse_min[k] = mse[i]
                r2_for_min_mse[k] = r2score[i]
                best_beta[k] = beta
                iteration_best = i

        mse_average[k] = np.mean(mse[k])
        r2score_average[k] = np.mean(r2score[k])
        bias_average[k] = np.mean(bias[k])    
        var_average[k] = np.mean(var[k])  

        # Run only for fifth degree:
        #fitted_patch = predict(rows, cols, beta)

        #surface_plot(fitted_patch,'Fitted terrain surface', 'Patch terrain suface',patch)
        #plt.show()

    return mse_average, r2score_average, bias_average, var_average, np.array(beta_list), best_beta, mse_min, r2_for_min_mse, iteration_best
