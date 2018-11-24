####################
# Project 3 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


import sys
import numpy as np
import scipy.sparse as sp
import warnings
import pickle
import os

#Comment this to turn on warnings
#warnings.filterwarnings('ignore')

np.random.seed(12)

# system size

process = input('Which dataset do you want to run, HIGGS or SUSY? [h/s]')


if dim == 'h':



elif dim == 's':
	NN = input('Do you want to run the Neural Network? [y/n].')
	num_classes = 1
	train_to_test_ratio = 0.7 # Training samples
	
	# path to data directory
	path_to_data = 'IsingData'
	file_name = "Ising2DFM_reSample_L40_T=All.pkl" # This file contains 16*10000 samples taken in the T=0.25 to T=4.00 temp range
	path = os.path.join("..",path_to_data, file_name)

	data = pickle.load(open(path,'rb')) # pickle reads the file, and returns the Python object (1D array,compressed bits)
	data = np.unpackbits(data).reshape(-1,1600) # Decompress array and reshape for convenience as a 40x40 lattice
	data = data.astype('int')
	data[np.where(data==0)] = -1 # map 0 state to -1 (Ising variable can take values +/-1)
	
	file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # This file contains 16*10000 samples taken in the above range
	path = os.path.join("..",path_to_data, file_name)
	labels = pickle.load(open(path,'rb'))


	del data, labels-

	# Define training and test data sets
	X = np.concatenate((X_ordered,X_disordered))
	Y = np.concatenate((Y_ordered,Y_disordered))

	X_train, X_test, Y_train, Y_test, = train_test_split(X,Y,train_size = train_to_test_ratio)



	TwoDim(X_train,X_test,Y_train,Y_test, NN, num_classes)

else:
	print("Choose a dataset, Higgs or SUSY.")




