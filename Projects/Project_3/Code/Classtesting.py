import numpy as np
import os
import pickle as pkl


# Importing data

def dataimport(which_set, benchmark, derived_feat=True, start=0, stop=np.inf):

	args = locals()

	# path to data directory
	path_to_data = 'Data'
	file_name = "Ising2DFM_reSample_L40_T=All.pkl" # This file contains 16*10000 samples taken in the T=0.25 to T=4.00 temp range
	path = os.path.join("..",path_to_data, file_name)