import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def Network(X, num_layers, num_nodes):

	model = tf.keras.Sequential()

	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes[i], activation='relu', input_dim=X.shape[1]))
		model.add(tf.keras.layers.Dropout(0.2))


