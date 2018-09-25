import numpy as np
from sklearn.linear_model import LinearRegression
x = np.random.randn(100)
y = np.random.randn(100)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y)

#z = 2*x + y

X = np.c_[np.ones(100), x**2, x, x*y, y, y**2]

beta = np.linalg.pinv(X).dot(z)

zpredict = X.dot(beta)


mse = 1.0/z.shape[0] *np.sum((z - zpredict)**2)
print(mse)


linreg = LinearRegression()

linreg.fit(X, z)

zpredict_sk = linreg.predict(X)

mse_sk = 1.0/z.shape[0] *np.sum((z - zpredict_sk)**2)
print(mse_sk)