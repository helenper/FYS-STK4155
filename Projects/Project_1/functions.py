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
    #print(np.shape(X))
    #print('z',np.shape(z))
    #beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z) 
    beta = (np.linalg.pinv(X.T @ X)@ X.T @ z)
    #print('beta', np.shape(beta))
    #print(np.shape(beta))
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
    #print(np.shape(beta_ridge))

    pred_ridge =  X_test @ beta_ridge 
    
    mse, R2, bias, variance = quality(z_test, pred_ridge)
    #print(np.shape(beta_ridge))
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

def betaConfidenceInterval(beta, best_beta, iteration_best):
    confidenceInterval = []
    for i in range(best_beta.shape[0]):
        sigma = np.sqrt(np.var(beta[iteration_best][i]))
        confidenceInterval_start = np.mean(best_beta[i]) - 2*sigma
        confidenceInterval_end = np.mean(best_beta[i]) + 2*sigma
        confidenceInterval.append([confidenceInterval_start, confidenceInterval_end])
    
    return confidenceInterval


'''
    beta_var = np.zeros((beta.shape[1], beta.shape[0]*beta.shape[2]))
    for i in range(beta.shape[1]):
        for j in range(beta.shape[0]):
            beta_var[i] = beta[i][j]
            print(beta_var[i])
'''
  

def runFranke(polydegree, lambda_values, num_data, num_iterations,seed, method):
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
    #x = C
    #y = R

    #print('FrankeFunction - x', np.shape(x))

    z = FrankeFunction(x,y) + noise*np.random.randn(len(x))
    #print('FrankeFunction', np.shape(z))
    


    #---------------------------------------------------------------------
    # Use bootstrap to define train and test data and calculate a mean 
    # value for MSE and R2 for the different methods OSL, Ridge and Lasso
    #---------------------------------------------------------------------

    iterations = num_iterations    # number of times we split and save our calculations in train and test point

    # Create arrays to hold different values to be taken mean over later. 
    # Each arrray is a nested array, where the first index points to the degree of the polynomial
    # used in that iteration. 
    X = polynomialfunction(x,y,len(x),degree=polydegree)
    #X = np.hstack(X)
    #print(np.shape(X))
    #print('Franke', np.shape(X))
    #indices = np.array([i for i in range(len(z))])
    X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)

    #, indices_train, indices_test
    #mse = np.zeros((len(lambda_values), iterations))
    #r2score = np.zeros((len(lambda_values), iterations))
    #bias = np.zeros((len(lambda_values), iterations))
    #var = np.zeros((len(lambda_values), iterations))
    #beta = np.zeros((len(lambda_values), iterations))

    #mse_average = np.zeros((len(lambda_values),1))
    #r2score_average = np.zeros((len(lambda_values), 1))
    #bias_average = np.zeros((len(lambda_values), 1))
    #var_average = np.zeros((len(lambda_values), 1))
    #beta = np.zeros((len(lambda_values), 1))

    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta_list = [] #np.zeros(iterations)

    beta= 0
    best_beta = 0

    mse_min = 1000
    r2_for_min_mse = 0
    #best_beta = np.zeros(X.shape[0])
    k =0
    

    #for k in range(len(lambda_values)):
    for i in range(iterations):
        X_train, z_train = bootstrap(X_train,z_train)
        #print('X_train', np.shape(X_train))
        if method == 'OLS':
            mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
            beta_list.append(beta)
            #print(np.shape(beta))
        #if method == ridge:
        #    mse.append(ridge(X_train,z_train,X_test,z_test,lmd)[0])

        if method == 'Ridge':
            #mse[k][i], r2score[k][i], bias[k][i], var[k][i] = ridge(X_train,z_train,X_test,z_test,lambda_values[k])
            #print(mse)
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
        
            #best_beta = []
            #best_beta.append(beta)

        # Average qualities:
        #mse_average[k] = np.mean(mse[k])
        #r2score_average[k] = np.mean(r2score[k])
        #bias_average[k] = np.mean(bias[k])    
        #var_average[k] = np.mean(var[k])  

    mse_average = np.mean(mse[k])
    r2score_average = np.mean(r2score[k])
    bias_average = np.mean(bias[k])    
    var_average = np.mean(var[k])  

    #print('b-list:', np.shape(beta_list))

    return mse_average, r2score_average, bias_average, var_average, np.array(beta_list), best_beta, mse_min, r2_for_min_mse, iteration_best

       

def runTerrain(polydegree, lambda_values, n, iterations, method = OLS, seed=False):
    terrain1 = imread('terrainone.tif')
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

    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta_list = [[] for i in range(len(lambda_values))]
    print(beta_list)
    mse_min = 10000
    best_beta = 0


    for k,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
        row_end = row_start + patch_size_row
        col_end = col_start + patch_size_col

        patch = terrain1[row_start:row_end, col_start:col_end]

        z = patch.reshape(-1,1)
        print('blaaa', np.shape(z))
        X = polynomialfunction(x,y,len(x),degree=polydegree)
        #X = X.reshape(-1,1)
        print('terrain', np.shape(X))
        X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)
        #print('hei',np.shape(X_train))
        for lmd in lambda_values:
            for i in range(iterations):
                X_train, z_train = bootstrap(X_train, z_train)
                #print('X_train', np.shape(X_train))
                if method == OLS:
                    mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,z_train, X_test, z_test)
                    print(mse)
                    #print(np.shape(beta))
                if method == ridge:
                    mse[i], r2score[i], bias[i], var[i],beta = ridge(X_train,z_train,X_test,z_test,lmd)
                if method == lasso:
                    mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,z_train,X_test,z_test,lmd)

                """
                if mse[i] < mse_min: 
                    mse_min = mse[i]
                    #print(mse_min)
                    r2_for_min_mse = r2score[i]
                    best_beta = []
                    #print(np.shape(beta))
                    for j in range(len(beta)):
                        best_beta.append(beta[j][i])
                # Getting beta for the confidence interval
                #betaConfidenceInterval(beta, beta_file)
                """
                mse_average = np.mean(mse[k])
                print(mse_average) 
                #print(mse_average) 
        r2score_average = np.mean(r2score[k])
        bias_average = np.mean(bias[k])    
        var_average = np.mean(var[k])


        #mse_average = [mse_average[0], mse_average[1], mse_average[2], mse_average[3], mse_average[4]]
        #print(mse_average) 
    """
    mse_average = np.mean(mse)  
    r2score_average = np.mean(r2score)
    bias_average = np.mean(bias)    
    var_average = np.mean(var)  
    """

    return mse_average, r2score_average, bias_average, \
                var_average, beta, best_beta, mse_min
