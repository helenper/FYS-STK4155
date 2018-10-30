import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(0)


def sigmoid(X):
	return 1/(1 + np.exp(-X))



def feed_forward(X_train,E_train, h_weights, h_bias, o_weights, o_bias):
	# weighted sum of inputs to the hidden layer
	z_h = np.matmul(X_train,h_weights) + h_bias
	a_h = sigmoid(z_h)
	z_o = np.matmul(a_h, o_weights) + o_bias
	print(z_o.shape, o_weights.shape)
	exp_term = np.exp(z_o)
	probs = exp_term/np.sum(exp_term, axis=1, keepdims=True)
	a_o = sigmoid(z_o)
	return a_h, probs, a_o, z_o


def backwardpropagation(X_train,E_train, h_weights, h_bias, o_weights, o_bias):
	a_h, probs, a_o, z_o = feed_forward(X_train,E_train, h_weights, h_bias, o_weights, o_bias)
	error_output = a_o - E_train
	print(z_o, E_train)

	error_hidden = np.matmul(error_output,o_weights.T) * a_h * (1 - a_h)

	output_gradient_weights = np.matmul(a_h.T, error_output)
	output_gradient_bias = np.sum(error_output, axis=0)

	hidden_gradient_weights = np.matmul(X.T, error_hidden)
	hidden_gradient_bias = np.sum(error_hidden, axis=0)

	return output_gradient_weights, output_gradient_bias, 	hidden_gradient_weights, hidden_gradient_bias


def Neural_Network(X_train, E_train, num_classes, m, lmbd=1):
	n_inputs, n_features = X_train.shape
	n_h_neurons = 50
	n_categories = 1

	h_weights = np.random.randn(n_features, n_h_neurons)
	h_bias = np.zeros(n_h_neurons) + 0.01

	o_weights = np.random.randn(n_h_neurons, n_categories)
	o_bias = np.zeros(n_categories) + 0.01
	
	eta = 0.01

	for i in range(1000):
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

	return o_weights

def Neural_Network_Classification():
	n_inputs, n_features = X_train.shape
	n_h_neurons = 50
	n_categories = 

	h_weights = np.random.randn(n_features, n_h_neurons)
	h_bias = np.zeros(n_h_neurons) + 0.01

	o_weights = np.random.randn(n_h_neurons, n_categories)
	o_bias = np.zeros(n_categories) + 0.01
	
	eta = 0.01
	for i in range(1000):
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

	return o_weights
	return 0