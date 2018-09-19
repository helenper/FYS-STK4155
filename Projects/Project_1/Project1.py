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

#def OLS(x, y, z): 
    

# Make the dataset
n = 100							# number of datapoints
x = np.random.uniform(0.0,1.0, n)		# create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)		# create a random number for y-values in dataset with noise

x_, y_ = np.meshgrid(x,y)	# Gives all combinations of x and y in two matrices
#print(np.shape(x), np.shape(y)) 

#print(x)
x = x_.reshape(-1,1) # Reshape matrix to be a 1 coloum matrix
y = y_.reshape(-1,1) # Reshape matrix to be a 1 coloum matrix
#print(np.shape(x), np.shape(y))
num_rows = n 	# Choose this to be number of x points
num_cols = n 	# Choose this to be number of y points

xb= np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \

                x**3, x**2*y, x*y**2, y**3, \

                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \

                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]



z = FrankeFunction(x, y)
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z) 
RSS = (y-xb.dot(beta)).T.dot((y - xb.dot(beta))) 

#print(x.flatten())
#x = np.sort(x.flatten())
#print(type(x))
#y = np.sort(y.flatten())
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

x_, y_ = np.meshgrid(x,y)
z = FrankeFunction(x_, y_) +0.9*np.random.randn(1)

zpredict = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        x_value = x[i]
        y_value = y[j]
        zpredict[j,i] = np.array([1 , x_value, y_value, x_value**2, x_value*y_value, y_value**2, \

                x_value**3, x_value**2*y_value, x_value*y_value**2, y_value**3, \

                x_value**4, x_value**3*y_value, x_value**2*y_value**2, x_value*y_value**3,y_value**4, \

                x_value**5, x_value**4*y_value, x_value**3*y_value**2, x_value**2*y_value**3,x_value*y_value**4, y_value**5]).dot(beta)

"""
for alle (xverdi,yverdi) i x, y
    zpredict = [1, x, y, x**2, x*y, y**2, ...]*beta
"""
#print(z)
# OLS 

#x = x.reshape(n,n)
#y= y.reshape(n,n)
#z = z.reshape(n,n)
#print(z.shape)
"""
polyreg = PolynomialFeatures(degree=5)
xb = polyreg.fit_transform(x)
linreg = LinearRegression()
linreg.fit(xb,y)
zpredict_mse = linreg.predict(xb)
"""
"""
# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_, y_, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


#plt.show()

#Plotting our prediction
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_, y_, zpredict, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""

# The mean squared error  

def MSE(y, y_tilde):
    mse_calc = 0
    for i in range(len(y)):
        mse_calc += (y[i] - y_tilde[i])**2
    return mse_calc/len(y)


z = z.reshape(-1,1)
zpredict = zpredict.reshape(-1,1)
print('Mean squared error: %.5f' % MSE(z, zpredict))
print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict))


# Explained variance score: 1 is perfect prediction      
def R_2(y, y_tilde):
    y_mean = np.mean(y)
    r2_calc_up = 0
    r2_calc_down = 0
    for i in range(len(y)):
        r2_calc_up += (y[i] - y_tilde[i])**2
        r2_calc_down += (y[i]-y_mean)**2
    r2_calc = r2_calc_up/r2_calc_down
    return 1-r2_calc

print('Variance score: %.2f' % R_2(z, zpredict))
print('Variance score scitkitlearn: %.2f' % r2_score(z, zpredict))

#print()

