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
from Classtesting import *
#Comment this to turn on warnings
#warnings.filterwarnings('ignore')

np.random.seed(12)

# system size

data = input('Which dataset do you want to run, HIGGS or SUSY? [h/s]')


if data == 'h':

	X_train, y_train = dataimport(data, train)
	X_validate, y_validate = dataimport(data, validate)
	X_test, y_test = dataimport(data, test)

elif data == 's':

	X_train, y_train = dataimport(data, train)
	X_validate, y_validate = dataimport(data, validate)
	X_test, y_test = dataimport(data, test)


else:
	print("Choose a dataset, Higgs or SUSY.")




