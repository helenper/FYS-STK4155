import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import time

def Network(X_train, y_train, X_validate, y_validate, X_test, y_test, num_layers, num_nodes, batch_size, epochs, data, drop, input_hidden_activation, output_activation, derived_feat):

	model = tf.keras.Sequential()

	start_time = time.time()

	# Weights initializers for the different layers

	input_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1) 
	hidden_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.05)
	output_initializer = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.001)

	model.add(tf.keras.layers.Dense(num_nodes, kernel_initializer=input_initializer, activation=input_hidden_activation, input_dim=X_train.shape[1]))
	
	if drop == 'True': model.add(tf.keras.layers.Dropout(0.3))
	
	for i in range(num_layers):
		model.add(tf.keras.layers.Dense(num_nodes, kernel_initializer=hidden_initializer, activation=input_hidden_activation))
		if drop == 'True': model.add(tf.keras.layers.Dropout(0.3))	

	model.add(tf.keras.layers.Dense(y_train.shape[1], kernel_initializer=output_initializer, activation=output_activation))
	

	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=10, verbose=1)

	#exp_learning_rate = tf.keras.callbacks.LearningRateScheduler()

	sgd = tf.keras.optimizers.SGD(lr=0.05,momentum=0.95,decay=0.0000002) 
	model.compile(optimizer=sgd, loss='binary_crossentropy')


	model_info = model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=[X_validate,y_validate], callbacks=[earlystop])#,exp_learning_rate])


	ypred = model.predict(X_test, batch_size=batch_size)

	print('Layers: ', num_layers, 'Nodes: ', num_nodes, 'Batch size: ', batch_size)

	AUC = roc_auc_score(y_test,ypred)
	False_positive_rate, True_positive_rate, b = roc_curve(y_test,ypred)

	print('AUC: ', AUC)

	file = open('AUC_result_layers%s_nodes%s_batch%s_%s_drop%s_Feat%s.txt' % (num_layers,num_nodes,batch_size,data,drop,derived_feat),'w')
	file.write('AUC: %f \n' % AUC)
	file.write('Dataset: %s \n' % data)
	file.write('Nodes: %f \n' % num_nodes)
	file.write('Batch: %f \n' % batch_size)
	file.write('Layers: %f \n' % num_layers)
	file.write('Total runtime: %f' % (time.time() - start_time))
	file.close()

	"""
	plt.plot(model_info.history['val_loss'])#range(1,len(model_info.history['val_loss']) + 1), model_info.history['val_loss'])
	plt.title('Model Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Val Loss')
	#plt.xticks(np.arange(1,len(model_info.history['val_loss'])+1),len(model_info.history['val_loss'])/10)
	plt.savefig('Loss_dataset_%s_nodes%f_nlayers%f.txt' % (data,num_nodes,num_layers))
	"""
	
	#plt.figure()

	plt.plot(True_positive_rate,False_positive_rate)
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	plt.xlabel('True positive rate')
	plt.ylabel('False positive rate')
	plt.title('ROC Curve')
	plt.savefig('ROC_dataset_%s_nodes%f_nlayers%f_%s_Feat%s.txt' % (data,num_nodes,num_layers,drop,derived_feat))
	
