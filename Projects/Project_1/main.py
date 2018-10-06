####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
from plotfunctions import *

lambda_values = [1e-2]#, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

n = 100
iterations = 10

mse_OLS_average, r2score_OLS_average,\
bias_OLS_average, var_OLS_average, beta_OLS, best_beta, \
min_mse = runFranke(5, lambda_values,n, iterations, method=lasso, seed=True)


print(mse_OLS_average,  r2score_OLS_average, bias_OLS_average,var_OLS_average)

#print(np.shape(best_beta), min_mse)

#betaConfidenceInterval(beta_OLS, best_beta)
#plotMSE(mse_OLS_average)

"""
mse = np.sum( (fitted_patch - patch)**2 )/num_data
R2 = 1 - np.sum( (fitted_patch - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
var = np.sum( (fitted_patch - np.mean(fitted_patch))**2 )/num_data
bias = np.sum( (patch - np.mean(fitted_patch))**2 )/num_data
"""