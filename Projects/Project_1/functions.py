####################
# Project 1 - functions 
# FYS-STK 3155/4155
# Fall 2018 
####################

# Ignore warnings
import warnings
warnings.simplefilter("ignore")

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

    #beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 
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

    n_samples = 100

    IX = np.eye(X.shape[1])

    beta_ridge = (np.linalg.pinv( X.T @ X + lambda_value*IX) @ X.T @ z) 

    pred_ridge =  X_test @ beta_ridge 
    
    mse, R2, bias, variance = quality(z_test, pred_ridge)
    return mse, R2, bias, variance, beta_ridge


def lasso(X,z,X_test, z_test, lambda_value):
    ''' A function that implements the Lasso method'''

    lasso=Lasso(lambda_value, max_iter=1e6, normalize = True, fit_intercept = False)
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

def betaConfidenceInterval(beta, best_beta):
    sigma = np.zeros(len(beta))
    sigma = np.sqrt(np.var(beta))
    confidenceInterval_start = best_beta-2*sigma
    confidenceInterval_end = best_beta+2*sigma
    print(confidenceInterval_start, confidenceInterval_end)
    

def runFranke(polydegree, lambda_values, num_data, num_iterations, method = OLS, seed=False):
    if seed== True:
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

    # Create arrays to hold different values to be taken mean over later. 
    # Each arrray is a nested array, where the first index points to the degree of the polynomial
    # used in that iteration. 
    X = polynomialfunction(x,y,len(x),degree=polydegree)
    indices = np.array([i for i in range(len(z))])
    X_train, X_test, z_train, z_test, indices_train, indices_test = train_test_split(X, z, indices, train_size = 0.7)

    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta = np.zeros(iterations)

    mse_min = 1000
    r2_for_min_mse = 0
    #best_beta = np.zeros(X.shape[0])

    for lmd in lambda_values:
        for i in range(iterations):
            X_train, z_train = bootstrap(X_train,z_train)
            if method == OLS:
                mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
            if method == ridge:
                mse[i], r2score[i], bias[i], var[i],beta = ridge(X_train,z_train,X_test,z_test,lmd)
            if method == lasso:
                mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,z_train,X_test,z_test,lmd)

            if mse[i] < mse_min: 
                mse_min = mse[i]
                r2_for_min_mse = r2score[i]
                best_beta = []
                for j in range(beta.shape[0]):
                    best_beta.append(beta[j][i])

        # Average qualities:
        mse_average = np.mean(mse)
        r2score_average = np.mean(r2score)
        bias_average = np.mean(bias)    
        var_average = np.mean(var)  


    return mse_average, r2score_average, bias_average, \
            var_average, beta, best_beta, mse_min


def runTerrain(polydegree, lambda_values, method = OLS, seed=False):
    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta = np.zeros(iterations)

    for k,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
        row_end = row_start + patch_size_row
        col_end = col_start + patch_size_col

        patch = terrain1[row_start:row_end, col_start:col_end]

        z = patch.reshape(-1,1)
        X = polynomialfunction(x,y,len(x),degree=polydegree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)
        for a in alpha:
            for i in range(iterations):
                X_train, z_train = bootstrap(X_train, z_train)
                if method == OLS:
                    mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
                if method == ridge:
                    mse[i], r2score[i], bias[i], var[i],beta = ridge(X_train,z_train,X_test,z_test,lmd)
                if method == lasso:
                    mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,z_train,X_test,z_test,lmd)

                # Getting beta for the confidence interval
                #betaConfidenceInterval(beta, beta_file)


                mse_average = np.mean(mse)  
                r2score_average = np.mean(r2score)
                bias_average = np.mean(bias)    
                var_average = np.mean(var)  


