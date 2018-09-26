import numpy as np
n = 10 
num_rows = n
num_cols = n

def polynomialfunction(x, y, type):
    if type==1: 
        X = np.c_[np.ones((num_rows*num_cols,1)) , x, y]

    elif type==2:
        X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2]

    elif type==3:
        X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3]

    elif type==4:
        X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4]

    elif type==5:
        X = np.c_[np.ones((num_rows*num_cols,1)) , x, y, x**2, x*y, y**2, \
                x**3, x**2*y, x*y**2, y**3, \
                x**4, x**3*y, x**2*y**2, x*y**3,y**4, \
                x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]
    else:
        print('Degree out of range!')

    return X

x = np.random.uniform(0.0,1.0, n*n)       
y = np.random.uniform(0.0,1.0, n*n)

X = polynomialfunction(x, y, type=4) # Give your wish for degree as type

print(np.shape(X))

