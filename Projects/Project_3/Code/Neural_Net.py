import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def Network(X, y, num_layers, num_nodes):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(num_nodes[i], activation='relu', input_dim=X.shape[1]))
	model.add(tf.keras.layers.Dropout(0.2))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes[i], activation='relu'))
		model.add(tf.keras.layers.Dropout(0.2))

	model.add(tf.keras.layers.Dense(y.shape[1], activation='softmax'))


	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	model.fit(X,y,epochs=100,batch_size=32,validation_data=[])

