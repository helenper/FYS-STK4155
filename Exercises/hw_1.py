import numpy as np 
from random import random, seed 
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

x = np.random.rand(100,1)
y = 5*x**2+0.1*np.random.randn(100,1)

#2a
xb = np.c_[np.ones((100,1)), x, x**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
print('beta=', beta)
xnew = np.random.random(size=(50, 1)) + 1
xbnew = np.c_[np.ones((50,1)), xnew, xnew**2]
ypredict = xbnew.dot(beta)

# plt.plot(xnew, ypredict, 'bo')
# plt.plot(x, y ,'ro')
# plt.axis([0,2.0,0, 15.0])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title(r'Polynomial Regression')
# plt.show()

#2b
polyreg = PolynomialFeatures(degree=2)
xb = polyreg.fit_transform(x)
linreg = LinearRegression()
linreg.fit(xb,y)
xnew = np.random.random(size=(100, 1)) + 1
xbnew = polyreg.fit_transform(xnew)
ypredict_ = linreg.predict(xbnew)

ypredict_mse = linreg.predict(xb)

# plt.plot(xnew, ypredict_, "bo")
# plt.plot(x, y ,'ro')
# plt.axis([0,2.0,0, 15.0])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title(r'Polynomial Regression with scikit-learn')
# plt.show()

#2c
print('The intercept alpha: \n', linreg.intercept_)
print('Coefficient beta : \n', linreg.coef_)
# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(y, ypredict_mse))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(y, ypredict_mse))
# Mean squared log error                                                        
print('Mean squared log error: %.2f' % mean_squared_log_error(y, ypredict_mse) )
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(y, ypredict_mse))