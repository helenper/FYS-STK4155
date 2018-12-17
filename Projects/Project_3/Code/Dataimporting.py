import numpy as np
import os
import pickle as pkl
import pandas as pd

# Importing data

def dataimport(which_set, derived_feat, start=0, stop=np.inf):

	args = locals()

	# path to data directory
	if which_set == 'h':
		file_name = "HIGGS.csv" 
		ntrain = 10000000
		nvalid = 500000
		ntest = 500000
	elif which_set == 's':
		file_name = "SUSY.csv"
		ntrain = 4000000
		nvalid = 500000
		ntest = 500000

	path_to_data = 'Data'
	path = os.path.join("..",path_to_data, file_name)

	X = pd.read_csv(path)	
	

	y = X.iloc[:,0]
	X = X.iloc[:,1:]

	X = np.array(X, dtype='float32')
	y = np.array(y, dtype='float32').reshape((-1,1))
	
	X_train = X[0:ntrain, :]
	y_train = y[0:ntrain]
	
	X_validate = X[ntrain:ntrain+nvalid, :]
	y_validate = y[ntrain:ntrain+nvalid]
	
	X_test = X[ntrain+nvalid:ntrain+nvalid+ntest, :]
	y_test = y[ntrain+nvalid:ntrain+nvalid+ntest]

	if which_set == 'h' and derived_feat == 'high':
		# Only the 7 high level features
		X_train = X_train[:, 21:28]
		X_validate = X_validate[:, 21:28]
		X_test = X_test[:, 21:28]
	elif which_set == 'h' and derived_feat == 'low':
		# Only the 21 raw features
		X_train = X_train[:, 0:21]
		X_validate = X_validate[:, 0:21]
		X_test = X_test[:, 0:21]
	elif which_set == 'h' and derived_feat == 'regress':
		# Predict high level features from low level
		y_train = X_train[:, 21:28]
		y_validate = X_validate[:, 21:28]
		y_test = X_test[:, 21:28]
		X_train = X_train[:, 0:21]
		X_validate = X_validate[:, 0:21]
		X_test = X_test[:, 0:21]
	elif which_set == 's' and derived_feat == 'high':
		# Only the 10 high level features
		X_train = X_train[:, 8:18]
		X_validate = X_validate[:, 8:18]
		X_test = X_test[:, 8:18]
	elif which_set == 's' and derived_feat == 'low':
		# Only the 8 low level features
		X_train = X_train[:, 0:8]
		X_validate = X_validate[:, 0:8]
		X_test = X_test[:, 0:8]

	return X_train, y_train, X_validate, y_validate, X_test, y_test


