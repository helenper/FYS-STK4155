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
import pickle
import os

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




elif dim == '2' or dim == 'two':
	num_classes = 2
	train_to_test_ratio = 0.7 # Training samples

	# path to data directory
	path_to_data = 'IsingData/'
	file_name = "Ising2DFM_reSample_L40_T=All.pkl" # This file contains 16*10000 samples taken in the T=0.25 to T=4.00 temp range
	
	data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file, and returns the Python object (1D array,compressed bits)
	data = np.unpackbits(data).reshape(-1,1600) # Decompress array and reshape for convenience as a 40x40 lattice
	data = data.astype('int')
	data[np.where(data==0)] = -1 # map 0 state to -1 (Ising variable can take values +/-1)
	
	file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # This file contains 16*10000 samples taken in the above range
	labels = pickle.load(open(path_to_data+file_name,'rb'))

	# divide data into ordered, critical and disordered
	X_ordered = data[:7000,:]
	Y_ordered = labels[:7000]

	X_critical = data[70000:100000,:]
	Y_critical = labels[70000:100000]

	X_disordered = data[100000:107000,:]
	Y_disordered = labels[100000:107000]

	del data, labels

	# Define training and test data sets
	X = np.concatenate((X_ordered,X_disordered))
	Y = np.concatenate((Y_ordered,Y_disordered))

	X_train, X_test, Y_train, Y_test, = train_test_split(X,Y,train_size = train_to_test_ratio)



	twodim = TwoDim(X_train,X_test,Y_train,Y_test)
#	for m in methods:
#		print('The next method to run is: %s' %m)



else:
	print("Must be 1 or 2 dimensions.")

states=np.random.choice([-1, 1], size=(10000,L))
energies=ising_energies(states,L)


