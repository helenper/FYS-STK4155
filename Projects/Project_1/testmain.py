####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
from plotfunctions import *

lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

n = 10
iterations = 4

mse_av, r2_av, bias_av, var_av = runFranke(5, lambda_values,n, iterations, method=ridge, seed=False)

print(r2_av)




'''
mse_OLS_average, r2score_OLS_average,\
bias_OLS_average, var_OLS_average = runFranke(5, lambda_values,n, iterations, method=ridge, seed=False)

print(mse_OLS_average)
'''

#betaConfidenceInterval(beta_OLS, best_beta)

#print('runTerrain')
#plotMSE(mse_OLS_average)
"""
mse_OLS_average, r2score_OLS_average,\
bias_OLS_average, var_OLS_average, beta_OLS, best_beta, \
min_mse = runTerrain(1, lambda_values,n, iterations, method=OLS, seed=True)

print(mse_OLS_average,r2score_OLS_average)
"""
