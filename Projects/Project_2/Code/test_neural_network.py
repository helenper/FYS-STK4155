import numpy as np 
import matplotlib as plt
import sklearn
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
np.random.seed(12)
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------

L = 40

states = np.random.choice([-1, 1], size = (10000, L))

def ising_energies(states,L):
	J = np.zeros((L,L), )
	for i in range(L):
		J[i, (i+1)%L]-= 1
	E = np.einsum('...i,ij,...j->...', states, J, states)
	return E

energies = ising_energies(states, L)[:,np.newaxis]

# Splitting up in train and test:
train_states, test_states, train_energies, test_energies = train_test_split(states, energies, train_size=0.7)


def sigmoid(x):
	return 1.0/(1+ np.exp(-x))


def feedforward(inputs, weights, bias):
	layer = inputs @ weigths[0] + bias[0]
	sigmoid_layer = sigmoid(layer)
	E = np.dot(sigmoid_layer, weigths[1]) + bias[1]

	return E, sigmoid_layer

def backpropagation(target, output, weigths, sigmoid_layer, bias, eta, inputs):
	error_out = output - target
	#print(np.shape(output), np.shape(target))
	#print(error_out.shape, weigths[1].shape, sigmoid_layer.shape)
	error_hidden = error_out @ weigths[1].T * sigmoid_layer*(1- sigmoid_layer)

	weigths[0] -= eta*inputs.T @ error_hidden
	weigths[1] -= eta*sigmoid_layer.T @ error_out

	bias[0] -= eta*np.sum(error_hidden, axis=0)
	bias[1] -= eta*np.sum(error_out, axis=0)
	return weigths, bias

N_hidden = 100
eta = 0.001

weigths1 = np.random.randn(train_states.shape[1], N_hidden)/100
weights2 = np.random.randn(N_hidden, 1)/100
weigths = np.array([weigths1, weights2])

bias1 = np.zeros(N_hidden) + np.random.random()/10
bias2 = np.random.random()/10
bias = np.array([bias1,bias2])

mse = []

batch = 100 

def quality(E_test,Epredict):
    '''A function that calculate the mean square error and the R2 score of 
    the values sendt in. If the write value is anything else than zero
    the function will print out the values'''

    # Mean squared error:
    #E_test = E_test.ravel()
    #Epredict = Epredict.ravel()
    mse = (1.0/(np.size(E_test))) *np.sum((E_test - Epredict)**2)
    #print("mse: ", mse)
    # Explained R2 score: 1 is perfect prediction 
    #R2 = 1- ((np.sum((E_test-Epredict)**2))/(np.sum((E_test-np.mean(E_test))**2)))
    # Bias:
    #bias = np.mean((E_test - np.mean(Epredict, keepdims=True))**2)
    # Variance:
    #variance = np.mean(np.var(Epredict, keepdims=True))
    
    #return mse, R2, bias, variance
    return mse

for i in range(10000):
	index = np.random.randint(7000, size = batch)
	E, sigmoid_layer = feedforward(train_states[index], weigths, bias)
	weigths, bias = backpropagation(train_energies[index], E, weigths, sigmoid_layer, bias, eta, train_states[index])
	mse.append(quality(energies[index], E))
#mse_average = np.mean(mse)
#print(mse_average)

test_E, repp = feedforward(test_states, weigths, bias)
print(quality(test_energies, test_E))

#print(max(abs(test_energies) - abs(test_E)))



