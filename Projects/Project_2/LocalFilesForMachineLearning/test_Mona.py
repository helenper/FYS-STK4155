import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(12)
L = 40

states = np.random.choice([-1, 1], size = (10000, L))

def ising_energies(states,L):
	J = np.zeros((L,L), )
	for i in range(L):
		J[i, (i+1)%L]-= 1
	E = np.einsum('...i,ij,...j->...', states, J, states)
	return E

#E = ising_energies(states, L)

# Splitting up in train and test:
#states_train, states_test, E_train, E_test = train_test_split(states, E, train_size=0.7)

n = 100
states=np.random.choice([-1, 1], size=(n,L)) # Make 10000 random states.

X = np.zeros((n,L*L))

for i in range(n):
    X[i] = np.outer(states[i],states[i]).ravel()


E=ising_energies(states,L) # Calculate the energies of the states.

X_train, X_test, E_train, E_test = train_test_split(X, E, train_size = 0.7)


def sigmoid(X):
	return 1/(1 + np.exp(-X))

def dsigmoid(X):
	return sigmoid(X)*(1-sigmoid(X))

def feed_forward(X_train, weights_hidden, bias_hidden, weights_output, bias_output):
	# 
	# weighted sum of inputs to the hidden layer
	#weights_hidden = wegths[0]
	#bias_hidden = bias[0]

	print(X_train.shape, weights_hidden.shape, bias_hidden.shape)
	z_hidden = np.matmul(X_train,weights_hidden).T + bias_hidden #hidden layer
	print(z_hidden.shape)
	activation_hidden = sigmoid(z_hidden) #Sigmoid layer hidden
	
	print(activation_hidden.shape)

	z_output = np.matmul(activation_hidden.T, weights_output).T + bias_output #layer
	#activation_output = sigmoid(z_output) # Sigmoid layer 

	activation_output = z_output
	print(activation_output.shape)


	#E = np.dot(activation_output, weights_output)

	return activation_hidden, activation_output, z_output, z_hidden


def backwardpropagation(X_train,E_train, weights_hidden, bias_hidden, weights_output, bias_output, activation_hidden, activation_output, z_output, z_hidden):
	# This function gives us new weights and biases and error outputs
	
	error_output = dsigmoid(z_output)*(activation_output-E_train.reshape(1,-1)) #(z_output - E_train.reshape(-1,1)) * activation_output * (1 - activation_output)

	print(weights_output.shape, error_output.shape, dsigmoid(z_hidden).shape)
	error_hidden = dsigmoid(z_hidden)*(weights_output@error_output) #np.matmul(error_output,weights_output.T) * activation_hidden * (1 - activation_hidden)

	output_gradient_weights = np.matmul(activation_hidden, error_output.T)
	output_gradient_bias = np.sum(error_output, axis=0)

	print(error_hidden.shape, X_train.T.shape, weights_hidden.shape)

	hidden_gradient_weights = np.matmul(X_train.T, error_hidden.T)
	hidden_gradient_bias = np.sum(error_hidden, axis=0)

	return output_gradient_weights, output_gradient_bias, 	hidden_gradient_weights, hidden_gradient_bias

def quality(E_test,Epredict):
    E_test = E_test.ravel()
    Epredict = Epredict.ravel()

    mse = (1.0/(np.size(E_test))) *np.sum((E_test - Epredict)**2)

    R2 = 1- ((np.sum((E_test-Epredict)**2))/(np.sum((E_test-np.mean(E_test))**2)))

    return mse, R2

def Neural_Network_1dim(X_train, E_train):


	n_inputs, n_features = X_train.shape
	n_h_neurons = 100
	n_categories = 1

	weights_hidden = np.random.randn(n_features, n_h_neurons)
	bias_hidden = np.zeros((n_h_neurons, 1)) + 0.01
	#bias_hidden = bias_hidden.reshape(-1,1)

	weights_output = np.random.randn(n_h_neurons, n_categories)
	bias_output = np.zeros((n_categories, 1)) + 0.01

	print('vuhu', weights_hidden.shape, bias_hidden.shape, weights_output.shape, bias_output.shape)
	
	eta = 0.0001
	#print(E_train)
	batch = 200

	for i in range(10000):
		# calculate gradients for bias and weights
		index = np.random.randint(len(X_train), size = batch)
		activation_hidden, activation_output, Epredict , z_hidden= feed_forward(X_train[index], weights_hidden, bias_hidden, weights_output, bias_output)
		dWo, dBo, dWh, dBh = backwardpropagation(X_train[index], E_train[index], weights_hidden, bias_hidden, weights_output, bias_output, activation_hidden, activation_output, Epredict, z_hidden)

		#update weights and biases
		weights_output -= eta * dWo
		weights_hidden -= eta * dWh
		print(dBo.shape, dWo.shape, dBh.shape, dWh.shape)
		bias_output -= eta * dBo
		bias_hidden -= eta * dBh

	return weights_output, weights_hidden, bias_output, bias_hidden

weights_output, weights_hidden, bias_output, bias_hidden = Neural_Network_1dim(X_train, E_train)

activation_hidden, activation_output, Epredict, z_hidden = feed_forward(X_test, weights_hidden, bias_hidden, weights_output, bias_output, z_hidden)
print(quality(E_test, Epredict))



#def Neural_Network_2dim()

