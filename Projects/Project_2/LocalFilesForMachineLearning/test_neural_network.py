import numpy as np 
import matplotlib as plt
import sklearn
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
#np.random.seed(12)
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


def feedforward(inputs, weigths, bias):
	layer = inputs @ weigths[0] + bias[0] #z_hidden
	sigmoid_layer = sigmoid(layer) #Activation
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

weigths1 = np.random.randn(train_states.shape[1], N_hidden)
weights2 = np.random.randn(N_hidden, 1)
weigths = np.array([weigths1, weights2])

bias1 = np.zeros(N_hidden) + np.random.random()
bias2 = np.random.random()
bias = np.array([bias1,bias2])

mse = []

batch = 100 

def quality(E_test,Epredict):

    mse = (1.0/(np.size(E_test))) *np.sum((E_test - Epredict)**2)

    return mse

for i in range(10000):
	index = np.random.randint(len(train_states), size = batch)
	E, sigmoid_layer = feedforward(train_states[index], weigths, bias)
	weigths, bias = backpropagation(train_energies[index], E, weigths, sigmoid_layer, bias, eta, train_states[index])
	mse.append(quality(energies[index], E))
#mse_average = np.mean(mse)
#print(mse_average)

test_E, repp = feedforward(test_states, weigths, bias)
print(quality(test_energies, test_E))

#print(max(abs(test_energies) - abs(test_E)))



