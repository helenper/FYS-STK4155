####################
# Project 1 
# FYS-STK 3155/4155
# Fall 2018 
####################


####################
# Franke function - given in exercise
####################

# Import necessary packages

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
fig = plt.figure()
ax = fig.gca(projection='3d')

'''
# Removed to be rewritten in part a

# Make data.
x = np.arange(0, 1, 0.05)	
y = np.arange(0, 1, 0.05)	
x, y = np.meshgrid(x,y)	
'''

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


'''
#z = FrankeFunction(x, y) 	# We dont want to call this function now, only example

#xb = np.c_[np.ones((100,1)), x, x**2]
#beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)


polyreg = PolynomialFeatures(degree=2)
xb = polyreg.fit_transform(x)
linreg = LinearRegression()
linreg.fit(xb,y)
ypredict_mse = linreg.predict(xb)
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

print("Mean squared error: %.2f" % mean_squared_error(z, ypredict_mse))
'''

####################
# Part a) - Ordnary Least Square on the Franke function with resampling
####################

def OLS(): 
	'''
	bla bla bla
	'''

# Make the dataset
n = 2							# number of datapoints
x = np.random.uniform(0.0,1.0, n)		# create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)+0.9*np.random.randn(1)		# create a random number for y-values in dataset with noise

x, y = np.meshgrid(x,y)	# Gives all combinations of x and y in two matrices 

x = x.reshape(-1,1) # Reshape matrix to be a 1 coloum matrix
y = y.reshape(-1,1) # Reshape matrix to be a 1 coloum matrix

num_rows = n 	# Choose this to be number of x points
num_cols = n 	# Choose this to be number of y points

xb= np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \

                x**3, x**2*y, x*y**2, y**3, \

                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \

                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]



beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y) 

RSS = (y-xb.dot(beta)).T.dot((y - xb.dot(beta))) 

z = FrankeFunction(x, y) 

'''
polyreg = PolynomialFeatures(degree=5)
xb = polyreg.fit_transform(x)
linreg = LinearRegression()
linreg.fit(xb,y)
ypredict_mse = linreg.predict(xb)
'''

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# The mean squared error  
print("Mean squared error: %.2f" % mean_squared_error(z, ypredict_mse))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(y, ypredict_mse))

print()

