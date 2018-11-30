import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def Network(X_train, y_train, X_validate, y_validate, X_test, y_test, num_layers, num_nodes, batch_size, epochs):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(num_nodes, activation='relu', input_dim=X_train.shape[1]))
	#model.add(tf.keras.layers.Dropout(0.3))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes, activation='relu'))
		#model.add(tf.keras.layers.Dropout(0.3))

	model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

	#sgd = tf.keras.optimizers.SGD(lr=1)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


	model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_validate,y_validate])


	ypred = model.predict(X_test, batch_size=batch_size)

	print('AUC: ', roc_auc_score(y_test,ypred))

	#print('Accuracy: ', model.evaluate(X_test, y_test))