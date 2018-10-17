####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
#from plotfunctions import *
import sys
import numpy as np
import scipy.sparse as sp
import warnings

#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(12)

# system size
L=40
iterations = 100
methods = ['OLS', 'Ridge', 'Lasso']
lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
dim = input('Which dimension do you want to run for? If one => write 1 or one, if two => write 2 or two.')

if dim == '1' or dim == 'one':
	for m in methods:
		print('The next method to run is: %s' %m)
		answer = input('Do you want to run for %s [y/n] ?' %m)
		if answer == 'y' or answer == 'Y':
			#file = open('results_OneDim_%s.txt' %m,  'w')
			mse_average, r2score_average, bias_average, var_average, beta, mse_min, R2_for_Min_MSE_value = OneDim(L, iterations, lambda_values, method=m)

			"""
			file.write('The results from running with lamda = %f \n' % lambda_values)
			file.write('MSE_average:        %f \n' %mse_average )
			file.write('R2_score_average:   %f \n' %r2score_average)
			file.write('Bias_average:       %f \n' %bias_average)
			file.write('Variance_average:   %f \n' %var_average)
			file.write('Min_MSE_value:      %f \n' %mse_min)
			file.write('R2_for_Min_MSE_value:       %f \n' %r2_min)
			file.write('The_best_beta_parameters:   %s \n' %the_beta_values)
			file.write('\n')
			file.close()
			"""
		if answer == 'n' or answer == 'N':
			print('Moving on')


methods = []

elif dim == '2' or dim == 'two':
	for m in methods:
		print('The next method to run is: %s' %m)
		


else:
	print("Must be 1 or 2 dimensions.")

states=np.random.choice([-1, 1], size=(10000,L))
energies=ising_energies(states,L)


