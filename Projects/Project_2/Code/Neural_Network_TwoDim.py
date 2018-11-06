import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#np.random.seed(0)


def sigmoid(z):
	return 1/(1 + np.exp(-z))  # Ok :)

def feed_forward(X_train, weights_hidden, bias_hidden, weights_output, bias_output):
	# 
	# weighted sum of inputs to the hidden layer
	z_hidden = np.matmul(X_train,weights_hidden) + bias_hidden #hidden layer
	activation_hidden = sigmoid(z_hidden) #Sigmoid layer hidden
	
	z_output = np.matmul(activation_hidden, weights_output) + bias_output #layer
	activation_output = sigmoid(z_output) # Sigmoid layer 

	return activation_hidden, activation_output


def backwardpropagation(X_train,E_train, weights_hidden, bias_hidden, weights_output, bias_output, activation_hidden, activation_output):
	#activation_hidden, activation_output = feed_forward(X_train, weights_hidden, bias_hidden, weights_output, bias_output)
	
	error_output = activation_output - E_train.reshape(-1,1)
	
	error_hidden = np.matmul(error_output,weights_output.T) * activation_hidden * (1 - activation_hidden)
	
	acc_during = Accuracy(error_output)
	
	output_gradient_weights = np.matmul(activation_hidden.T, error_output)
	output_gradient_bias = np.sum(error_output, axis=0)

	hidden_gradient_weights = np.matmul(X_train.T, error_hidden)
	hidden_gradient_bias = np.sum(error_hidden, axis=0)

	return output_gradient_weights, output_gradient_bias, hidden_gradient_weights, hidden_gradient_bias, acc_during


def Neural_Network_TwoDim(X_train, E_train,X_test, E_test, m, lmbd=1):
	n_inputs, n_features = X_train.shape
	n_h_neurons = 100
	n_categories = 1

	weights_hidden = np.random.randn(n_features, n_h_neurons)
	bias_hidden = np.zeros(n_h_neurons) + 0.01

	weights_output = np.random.randn(n_h_neurons, n_categories)
	bias_output = np.zeros(n_categories) + 0.01
	
	eta = 1e-1
	batch = 200
	
	activation_hidden, activation_output = feed_forward(X_test, weights_hidden, bias_hidden, weights_output, bias_output)
	error_output = 	activation_output - E_test.reshape(-1,1)
	Acc_before_train = Accuracy(error_output)
	Acc_training = []
		
	for i in range(3000):
		# calculate gradients
		index = np.random.randint(len(X_train), size = batch)

		activation_hidden, activation_output = feed_forward(X_train[index], weights_hidden, bias_hidden, weights_output, bias_output)
		dWo, dBo, dWh, dBh, acc = backwardpropagation(X_train[index], E_train[index], weights_hidden, bias_hidden, weights_output, bias_output, activation_hidden, activation_output)
		Acc_training.append(acc)
#		dWo, dBo, dWh, dBh = backwardpropagation(X_train, E_train, weights_hidden, bias_hidden, weights_output, bias_output)
		
		#update weights and biases
		weights_output -= eta * dWo
		weights_hidden -= eta * dWh
		bias_output -= eta * dBo
		bias_hidden -= eta * dBh

	activation_hidden, activation_output = feed_forward(X_test, weights_hidden, bias_hidden, weights_output, bias_output)
	error_output = 	activation_output - E_test.reshape(-1,1)
	Acc_after_train = Accuracy(error_output)

	Plot_Accuracy(Acc_training)




def Accuracy(error_output):

	correct = 0

	for i in range(len(error_output)):
		if error_output[i] < 0.5:
			correct += 1.0

	Acc = correct/(len(error_output))*100
	print("Accuracy percentage: ", Acc)
	return Acc


def Plot_Accuracy(acc):

	plt.plot(np.linspace(0,len(acc)-1, len(acc)) , acc, 'bo', markersize=2)
	plt.title("Accuracy for the neural network training on two dimensional Ising-model.")
	plt.xlabel("Number of iterations")
	plt.ylabel("Percentage of correct predictions")
	plt.show()