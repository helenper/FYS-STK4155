import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import time
import pylab

def Network(X_train, y_train, X_validate, y_validate, X_test, y_test, num_layers, num_nodes, batch_size, epochs, data, input_hidden_activation, output_activation, drop, derived_feat, optimizer, learning_rate):

	model = tf.keras.Sequential()

	start_time = time.time()


	input_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1) 
	hidden_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05)
	output_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.001)

	model.add(tf.keras.layers.Dense(num_nodes, kernel_initializer=input_initializer, activation=input_hidden_activation, input_dim=X_train.shape[1]))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes, kernel_initializer=hidden_initializer, activation=input_hidden_activation))
		if drop == 'True' and i == 0: model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(y_train.shape[1], kernel_initializer=output_initializer, activation=output_activation))
	

	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=10, verbose=1)


	sgd = tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.95,decay=0.0000002) 
	model.compile(optimizer=sgd, loss='binary_crossentropy') if optimizer == 'sgd' else model.compile(optimizer=optimizer, loss='binary_crossentropy')


	model_info = model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_validate,y_validate], callbacks=[earlystop])


	ypred = model.predict(X_test, batch_size=batch_size)

	print('Layers: ', num_layers, 'Nodes: ', num_nodes, 'Batch size: ', batch_size)

	AUC = roc_auc_score(y_test,ypred)
	False_positive_rate, True_positive_rate, b = roc_curve(y_test,ypred)

	print('AUC: ', AUC)

	file = open('AUC_result_layers%d_nodes%d_%s_drop-%s_Feat%s_optimizer-%s_lr-%.4f.txt' % (num_layers,num_nodes,data,drop,derived_feat,optimizer,learning_rate),'w')
	file.write('AUC: %f \n' % AUC)
	file.write('Dataset: %s \n' % data)
	file.write('Nodes: %f \n' % num_nodes)
	file.write('Batch: %f \n' % batch_size)
	file.write('Layers: %f \n' % num_layers)
	file.write('Dropout: %s \n' % drop)
	file.write('Feature: %s \n' % derived_feat)
	file.write('Optimizer: %s \n' % optimizer)
	file.write('Learning rate: %s \n' % learning_rate)
	file.write('Total runtime: %f' % (time.time() - start_time))
	file.close()


	plt.plot(True_positive_rate,1-False_positive_rate)
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	plt.xlabel('Signal efficiency', fontsize=18)
	plt.ylabel('Background rejection', fontsize=18)
	plt.title('ROC Curve',fontsize=20)
	pylab.xticks(fontsize=14)
	pylab.yticks(fontsize=14)
	plt.savefig('ROC_dataset_%s_nodes%d_nlayers%d_drop-%s_Feat%s_optimizer-%s_lr-%.4f.png' % (data,num_nodes,num_layers,drop,derived_feat,optimizer,learning_rate))
	
