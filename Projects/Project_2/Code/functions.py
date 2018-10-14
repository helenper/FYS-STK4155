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


def OLS(X, z, X_test, E_test):
    '''Calculate and return the z and Epredict value by 
    ordinary least squares method'''
    beta = (np.linalg.pinv(X.T @ X)@ X.T @ z)
    Epredict = X_test @ beta
    mse, R2, bias, variance = quality(E_test, Epredict) 
    return mse, R2, bias, variance, beta



def ridge(X, z, X_test, E_test, lambda_value):
    ''' A function that implementes the Rigde method'''
    IX = np.eye(X.shape[1])
    beta_ridge = (np.linalg.pinv( X.T @ X + lambda_value*IX) @ X.T @ z) 
    pred_ridge =  X_test @ beta_ridge 
    mse, R2, bias, variance = quality(E_test, pred_ridge)
    return mse, R2, bias, variance, beta_ridge


def lasso(X,z,X_test, E_test, lambda_value):
    ''' A function that implements the Lasso method'''

    lasso=Lasso(lambda_value, max_iter=1e7, normalize = True, fit_intercept = False)
    lasso.fit(X,z) 
    beta_lasso = lasso.coef_.T
    predl=lasso.predict(X_test)

    mse, R2, bias, variance = quality(E_test, predl)
    return mse, R2, bias, variance, beta_lasso

def quality(E_test,Epredict):
    '''A function that calculate the mean square error and the R2 score of 
    the values sendt in. If the write value is anything else than zero
    the function will print out the values'''

    # Mean squared error:
    mse = (1.0/(np.size(E_test))) *np.sum((E_test - Epredict)**2)
    # Explained R2 score: 1 is perfect prediction 
    R2 = 1- ((np.sum((E_test-Epredict)**2))/(np.sum((E_test-np.mean(E_test))**2)))
    # Bias:
    bias = np.mean((E_test - np.mean(Epredict, keepdims=True))**2)
    # Variance:
    variance = np.mean(np.var(Epredict, keepdims=True))
    
    return mse, R2, bias, variance

def bootstrap(x,y):

    indices = np.random.choice(len(y),len(y))
    x_train_new = x[indices]        
    y_train_new = y[indices]
    return x_train_new, y_train_new

 

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E 



def OneDim(L, iterations, lambda_values, method):

    states=np.random.choice([-1, 1], size=(10000,L)) # Make 10000 random states.
    
    energies=ising_energies(states,L) # Calculate the energies of the states.

    X_train, X_test, E_train, E_test = train_test_split(states, energies, train_size = 0.7)


    mse = np.zeros(iterations)
    r2score = np.zeros(iterations)
    bias = np.zeros(iterations)
    var = np.zeros(iterations)
    beta_list = [] 

    beta= 0


    file = open('results_OneDim_%s.txt' %method,  'w')
    if method == 'OLS':
        beta= 0
        best_beta = 0
        mse_min = 1000
        r2_for_min_mse = 0

        
        for i in range(iterations):
            X_train, E_train = bootstrap(X_train,E_train)
            print(i)
            mse[i], r2score[i], bias[i], var[i], beta = OLS(X_train,E_train, X_test, E_test)
            beta_list.append(beta)

            if mse[i] < mse_min: 
                mse_min = mse[i]
                r2_for_min_mse = r2score[i]
                best_beta = beta
                iteration_best = i

        mse_average = np.mean(mse)
        r2score_average = np.mean(r2score)
        bias_average = np.mean(bias)    
        var_average = np.mean(var)

        file.write('MSE_average:        %f \n' %mse_average)
        file.write('R2_score_average:   %f \n' %r2score_average)
        file.write('Bias_average:       %f \n' %bias_average)
        file.write('Variance_average:   %f \n' %var_average)
        file.write('Min_MSE_value:      %f \n' %mse_min)
        file.write('R2_for_Min_MSE_value:       %f \n' %r2_for_min_mse)
        file.write('\n') 
        file.close()

    else:

        mse_average = []
        r2score_average = []
        bias_average = []    
        var_average = []   
        mse_min = []
        iteration_best = []
        r2_for_min_mse = []

        for l, lambda_value in enumerate(lambda_values):
            if method == 'Ridge':
                for i in range(iterations):
                    X_train, E_train = bootstrap(X_train,E_train)
                    mse[i], r2score[i], bias[i], var[i], beta = ridge(X_train,E_train,X_test,E_test,lambda_value)
                    beta_list.append(beta)

            if method == 'Lasso':
                for i in range(iterations):
                    X_train, E_train = bootstrap(X_train,E_train)
                    print(i)
                    mse[i], r2score[i], bias[i], var[i], beta = lasso(X_train,E_train,X_test,E_test,lambda_value)
                    beta_list.append(beta)

            mse_average.append(np.mean(mse))
            r2score_average.append(np.mean(r2score))
            bias_average.append(np.mean(bias)) 
            var_average.append(np.mean(var))

            mse_min.append(min(mse))
            iteration_best.append(np.where(mse == mse_min[l]))
            print(mse)
            r2_for_min_mse.append(r2score[iteration_best[l]])
        print(r2_for_min_mse)

        [file.write('The results from running with lamda = %f \n' % lamb) for lamb in lambda_values]
        [file.write('MSE_average:        %f \n' %mse_ave) for mse_ave in mse_average]
        [file.write('R2_score_average:   %f \n' %r2score_ave) for r2score_ave in r2score_average]
        [file.write('Bias_average:       %f \n' %bias_ave) for bias_ave in bias_average]
        [file.write('Variance_average:   %f \n' %var_ave) for var_ave in var_average]
        [file.write('Min_MSE_value:      %f \n' %mse_min_val) for mse_min_val in mse_min]
        [file.write('R2_for_Min_MSE_value:       %f \n' %r2_min) for r2_min in r2_for_min_mse]
        file.write('\n')
        file.close()



    return mse_average, r2score_average, bias_average, var_average, np.array(beta_list), mse_min, r2_for_min_mse 

