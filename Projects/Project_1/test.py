from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



n = 100   
num_rows = n    # Choose this to be number of x points
num_cols = n    # Choose this to be number of y points

                      # number of datapoints
x = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset with noise

x_, y_ = np.meshgrid(x,y) 
x = x_.reshape(-1,1) # Reshape matrix to be a 1 coloum matrix
y = y_.reshape(-1,1)  # Gives all combinations of x and y in two matrices
z = FrankeFunction(x, y)

xb = np.c_[np.ones((10000,1)), x, y, x**2, x*y, y**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)
xnew = np.random.random(size=(10000, 1)) #+ 1
ynew = np.random.random(size=(10000,1)) #+1
xbnew = np.c_[np.ones((10000,1)), xnew,ynew, xnew*ynew, xnew**2, ynew**2]
zpredict = xb.dot(beta)
print(np.shape(zpredict))

#scitkitlearn
polyreg = PolynomialFeatures(degree=2)
xb = polyreg.fit_transform(x, y)
linreg = LinearRegression()
linreg.fit(xb,z)
xnew = np.random.random(size=(10000, 1)) + 1
xbnew = polyreg.fit_transform(xnew)
zpredict_ = linreg.predict(xbnew)

zpredict_mse = linreg.predict(xb)
print(np.shape(xb), np.shape(zpredict_mse))
#RSS = (z-xb.dot(beta)).T.dot((z - xb.dot(beta))) 



# The mean squared error  

def MSE(y, y_tilde):
    mse_calc = 0
    for i in range(len(y)):
        mse_calc += (y[i] - y_tilde[i])**2
    return mse_calc/len(y)


z = z.reshape(-1,1)
zpredict = zpredict.reshape(-1,1)

# Why should we use mse? 
#print('Mean squared error: %.5f' % MSE(z, zpredict_mse))
#print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict_mse))


# This gives equal answers
print('Mean squared error: %.5f' % MSE(z, zpredict))
print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict_mse))
#print("Mean squared error scikitlearn: %.5f" % mean_squared_error(z, zpredict))

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
print('Variance score scitkitlearn: %.2f' % r2_score(z, zpredict_mse))

#print()

