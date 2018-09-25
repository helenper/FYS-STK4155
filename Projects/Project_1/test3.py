from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Making data:
#x = np.arange(0,1,0.05)
#y = np.arange(0,1,0.05)
#x,y = np.meshgrid(x,y)

#Random data
n = 100
x = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset 
x = np.sort(x)
y = np.sort(y)
x,y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

"""
# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
"""

num_rows = n 	# Choose this to be number of x points
num_cols = n 	# Choose this to be number of y points
x1 = np.ravel(x)
y1 = np.ravel(y)
#z = z.reshape(-1,1)
z1 = np.ravel(z)
print(np.shape(x))

xyb= np.c_[np.ones((num_rows*num_cols,1)) , x1, y1, x1**2, x1*y1, y1**2]

beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z1) 


zpredict = np.zeros((n,n))#((len(x), len(y)))
for i in range(n):#len(x)):
    for j in range(n):#len(y)):
        x_value = x1[i]
        y_value = y1[j]
        zpredict[j,i] = np.array([1 , x_value, y_value, x_value**2, x_value*y_value, y_value**2]).dot(beta)

"""
# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, zpredict, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()
"""
zpredict = np.ravel(zpredict)


print(np.shape(zpredict))
print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z1, zpredict))
print('Variance score scitkitlearn: %.2f' % r2_score(z1, zpredict))




