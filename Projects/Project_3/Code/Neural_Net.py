import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def Network(X_train, y_train, X_validate, y_validate, X_test, y_test, num_layers, num_nodes, batch_size, epochs, data):

	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Dense(num_nodes, activation='tanh', input_dim=X_train.shape[1]))
	#model.add(tf.keras.layers.Dropout(0.3))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes, activation='tanh'))
		#model.add(tf.keras.layers.Dropout(0.3))

	model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

	sgd = tf.keras.optimizers.SGD(lr=0.05)
	model.compile(optimizer='sgd', loss='binary_crossentropy')


	model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_validate,y_validate])


	ypred = model.predict(X_test, batch_size=batch_size)

	print('Layers: ', num_layers, 'Nodes: ', num_nodes, 'Batch size: ', batch_size)

	AUC = roc_auc_score(y_test,ypred)

	print('AUC: ', AUC)

	file = open('AUC_result_layers%s_nodes%s_batch%s_%s.txt' % (num_layers,num_nodes,batch_size,data),'w')
	file.write('AUC: %f' % AUC)
	file.write('Dataset: %f' % data)
	file.write('Nodes: %f' % num_nodes)
	file.write('Batch: %f' % batch_size)
	file.write('Layers: %f' % num_layers)
	file.close()

	#print('Accuracy: ', model.evaluate(X_test, y_test))