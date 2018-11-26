import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def Network(X_train, y_train, X_validate, y_validate, X_test, y_test, num_layers, num_nodes):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(num_nodes, activation='relu', input_dim=X_train.shape[1]))
	model.add(tf.keras.layers.Dropout(0.2))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.2))

	model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))


	model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])


	model.fit(X_train,y_train,epochs=100,batch_size=32,validation_data=[X_validate,y_validate])


	print('Accuracy: ', model.evaluate(X_test, y_test))