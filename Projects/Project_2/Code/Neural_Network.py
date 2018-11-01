import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(0)


def sigmoid(z):
	return 1/(1 + np.exp(-z))  # Ok :)

def feed_forward(X_train, weights_hidden, bias_hidden, weights_output, bias_output):
	# 
	# weighted sum of inputs to the hidden layer
	z_hidden = np.matmul(X_train,weights_hidden) + bias_hidden #hidden layer
	activation_hidden = sigmoid(z_hidden) #Sigmoid layer hidden
	
	z_output = np.matmul(activation_hidden, weights_output) + bias_output #layer
	activation_output = sigmoid(z_output) # Sigmoid layer 

	return activation_hidden, activation_output, z_output


def backwardpropagation(X_train,E_train, weights_hidden, bias_hidden, weights_output, bias_output):
	activation_hidden, activation_output, z_output = feed_forward(X_train, weights_hidden, bias_hidden, weights_output, bias_output)
	error_output = z_output - E_train.reshape(-1,1)

	error_hidden = np.matmul(error_output,weights_output.T) * activation_hidden * (1 - activation_hidden)

	output_gradient_weights = np.matmul(activation_hidden.T, error_output)
	output_gradient_bias = np.sum(error_output, axis=0)

	hidden_gradient_weights = np.matmul(X_train.T, error_hidden)
	hidden_gradient_bias = np.sum(error_hidden, axis=0)

	return output_gradient_weights, output_gradient_bias, 	hidden_gradient_weights, hidden_gradient_bias


def Neural_Network(X_train, E_train, m, lmbd=1):
	n_inputs, n_features = X_train.shape
	n_h_neurons = 50
	n_categories = 1

	weights_hidden = np.random.randn(n_features, n_h_neurons)
	bias_hidden = np.zeros(n_h_neurons) + 0.01

	weights_output = np.random.randn(n_h_neurons, n_categories)
	bias_output = np.zeros(n_categories) + 0.01
	
	eta = 0.0001
	print(E_train)
	for i in range(10):
		# calculate gradients
		dWo, dBo, dWh, dBh = backwardpropagation(X_train, E_train, weights_hidden, bias_hidden, weights_output, bias_output)

		#update weights and biases
		weights_output -= eta * dWo
		weights_hidden -= eta * dWh
		bias_output -= eta * dBo
		bias_hidden -= eta * dBh

	return weights_output, weights_hidden, bias_output, bias_hidden

def Neural_Network_Classification(X_train, E_train, m):
	n_inputs, n_features = X_train.shape
	n_h_neurons = 50
	n_categories = 1

	h_weights = np.random.randn(n_features, n_h_neurons)
	h_bias = np.zeros(n_h_neurons) + 0.01

	o_weights = np.random.randn(n_h_neurons, n_categories)
	o_bias = np.zeros(n_categories) + 0.01
	
	eta = 0.01
	for i in range(10):
		# calculate gradients
		dWo, dBo, dWh, dBh = backwardpropagation(X_train, E_train, h_weights, h_bias, o_weights, o_bias)

		# regularization term gradients 
		if m == 'Ridge':
			dWo += lmdb * o_weights
			dWh += lmdb * h_weights

		#update weights and biases
		o_weights -= eta * dWo
		h_weights -= eta * dWh
		o_bias -= eta * dBo
		h_bias -= eta * dBh

	return 0
	