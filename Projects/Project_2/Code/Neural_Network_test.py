import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(0)

def sigmoid(X):
	return 1./(1+np.exp(-X))


def FeedForward(X_train, E_train, weights_hidden, bias_hidden, weights_output, bias_output):
	
	z_hidden = np.matmul(X_train, weights_hidden) + bias_hidden
	activation_hidden = sigmoid(z_hidden)

	z_output = np.matmul(activation_hidden, weights_output) + bias_output
	activation_output = sigmoid(z_output)

	probs = np.exp(z_output)/(np.sum(np.exp(z_output), axis=1, keepdims=True))

	return z_output, activation_output, activation_hidden, probs


def backwardpropagation(X_train, E_train, weights_hidden, bias_hidden, weights_output, bias_output):

	z_output, activation_output, activation_hidden, probs = FeedForward(X_train,E_train,weights_hidden,bias_hidden, weights_output,bias_output)
	print(z_output.shape, activation_output.shape, activation_hidden.shape, probs.shape, E_train.shape, weights_output.shape)

	error_output_reg = z_output - E_train.reshape(-1,1)
	print(error_output_reg.shape, weights_hidden.shape)
#	error_output_reg = (activation_output - E_train) * (1 - activation_output) * activation_hidden
	error_output_log = activation_output - E_train

	error_hidden_reg = np.matmul(error_output_reg, weights_output.T) * activation_hidden * (1 - activation_hidden)
#	error_hidden_log = 

	dWo = np.matmul(activation_hidden.T, error_output_reg)
	print(error_output_reg.T.shape, activation_hidden.shape, dWo.shape)
	dBo = np.sum(error_output_reg, axis=0)


	dWh = np.matmul(X_train.T, error_hidden_reg)
	dBh = np.sum(error_hidden_reg, axis=0)

	return dWo, dBo, dWh, dBh, z_output, activation_hidden, probs


def Neural_Network(X_train,E_train, num_classes, m, lmbd=0):
	number_of_inputs, number_of_features = X_train.shape
	number_of_hidden_neurons = 50
	number_of_categories = num_classes

	weights_hidden = np.random.randn(number_of_features, number_of_hidden_neurons)
	bias_hidden = np.zeros(number_of_hidden_neurons) + 0.01

	weights_output = np.random.randn(number_of_hidden_neurons, int(number_of_categories))
	bias_output = np.zeros(int(number_of_categories)) + 0.01

	eta = 0.01
	print(E_train.reshape(-1,1))
	for i in range(1000):

		dWo, dBo, dWh, dBh, z_output, activation_hidden, probs = backwardpropagation(X_train, E_train, weights_hidden, bias_hidden, weights_output, bias_output)

		if m == 'Ridge':
			dWo += lmbd*weights_output
			dWh += lmbd*weights_hidden


		weights_output -= eta * dWo
		weights_hidden -= eta * dWh
		bias_output -= eta * dBo
		bias_hidden -= eta * dBh
		print(i)
	print("dims: ", np.sum(weights_hidden, axis=1,keepdims=True))

	return np.sum(weights_hidden, axis=1,keepdims=True), probs
