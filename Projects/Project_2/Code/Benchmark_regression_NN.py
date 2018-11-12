import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import random

np.random.seed(12)
L = 40
n = 10000
states=np.random.choice([-1, 1], size=(n,L)) # Make 10000 random states.

X = np.zeros((n,L*L))

for i in range(n):
    X[i] = np.outer(states[i],states[i]).ravel()


def ising_energies(states,L):
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L]-=1.0

	E = np.einsum('...i,ij,...j->...',states,J,states)
	return E



energies=ising_energies(states,L) # Calculate the energies of the states.



X_train, X_test, E_train, E_test = train_test_split(X, energies, train_size = 0.7)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, activation='sigmoid', input_dim=X.shape[1]))
#model.add(tf.keras.layers.Dense(100, activation='sigmoid', input_dim=X.shape[1]))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])


model.fit(X_train, E_train, epochs=100, batch_size=200,validation_data=[X_test,E_test])


print('accuracy:', model.evaluate(X_test,E_test))

