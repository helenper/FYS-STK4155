import numpy as np
import os
import pickle as pkl


# Importing data

def dataimport(which_set, datatype, derived_feat=True, start=0, stop=np.inf):

	args = locals()

	# path to data directory
	if which_set == 'h':
		file_name = "HIGGS" 
		ntrain = 10000000
		nvalid = 500000
		ntest = 500000
	elif which_set == 's':
		file_name = "SUSY"
		ntrain = 4000000
		nvalid = 500000
		ntest = 500000

	
	path_to_data = 'Data'
	path = os.path.join("..",path_to_data, file_name)

	X = pkl.load(open(path, 'r'))

	y = X[:,0].reshape((-1,1))
	X = X[:,1:]
	X = np.array(X, dtype='float32')
	y = np.array(y, dtype='float32')

	if datatype == 'train':	
		X = X[0:ntrain, :]
		y = y[0:ntrain, :]
	elif datatype == 'validate':
		X = X[ntrain:ntrain+nvalid, :]
		y = y[ntrain:ntrain+nvalid, :]
	elif datatype == 'test':
		X = X[ntrain+nvalid:ntrain+nvalid+ntest, :]
		y = y[ntrain+nvalid:ntrain+nvalid+ntest, :]


	if which_set == 'h' and derived_feat == 'only':
		# Only the 7 high level features
		X = X[:, 21:28]
	elif which_set == 'h' and not derived_feat:
		# Only the 21 raw features
		X = X[:, 0:21]
	elif which_set == 'h' and derived_feat == 'regress':
		# Predict high level features from low level
		y = X[:, 21:28]
		X = X[:, 0:21]
	elif which_set == 's' and derived_feat == 'only':
		# Only the 10 high level features
		X = X[:, 8:18]
	elif which_set == 's' and not derived_feat:
		# Only the 8 low level features
		X = X[:, 0:8]




def standardize(X):


	for j in range(X.shape[1]):
		vec = X[:, j]
		if np.min(vec) < 0:
			# Assume data is Gaussian or uniform -- center and standardize
			vec = vec - np.mean(vec)
			vec = vec/np.std(vec)
		elif np.max(vec) > 1.0:
			# Assume data is exponential -- just set mean to 1
			vec = vec/np.mean(vec)

		X[:, j] = vec

	return X