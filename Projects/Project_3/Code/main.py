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
from Dataimporting import *
from Neural_Net import *

#Comment this to turn on warnings
#warnings.filterwarnings('ignore')

np.random.seed(12)

# system size

num_layers = 1
num_nodes = 300
batch_size = 100
epochs = 10

data = input('Which dataset do you want to run, HIGGS or SUSY? [h/s]')


X_train, y_train, X_validate, y_validate, X_test, y_test = dataimport(data,derived_feat=False)

Network(X_train,y_train,X_validate,y_validate,X_test,y_test,num_layers,num_nodes, batch_size, epochs, data)


