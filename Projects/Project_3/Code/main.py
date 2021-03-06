####################
# Project 3 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


import sys
import numpy as np
import warnings
import pickle
import os
from Dataimporting import *
from Neural_Net import *

#Comment this to turn on warnings
#warnings.filterwarnings('ignore')

#np.random.seed(12)

num_layers = np.int(sys.argv[1])

num_nodes =  np.int(sys.argv[2])

batch_size = np.int(sys.argv[3])

epochs = np.int(sys.argv[4])

data = sys.argv[5]



input_and_hidden_activation = sys.argv[6]
output_activation = sys.argv[7]

drop = sys.argv[8]

derived_feat = sys.argv[9]

optimizer = sys.argv[10]

Learning_rate = np.float(sys.argv[11])

X_train, y_train, X_validate, y_validate, X_test, y_test = dataimport(data,derived_feat=derived_feat)

Network(X_train,y_train,X_validate,y_validate,X_test,y_test,num_layers,num_nodes, batch_size, epochs, data, input_and_hidden_activation, output_activation, drop, derived_feat, optimizer, Learning_rate)


