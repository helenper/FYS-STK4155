from functions import *
from plotfunctions import *
from imageio import imread

#-------------------------------------------------
# Define a seed to use while testing the program. 
# Comment out this to have changing random numbers. 
np.random.seed(4155)
#-------------------------------------------------


#---------------------------------------------------
# Make test-dataset and calculate Franke's function
#---------------------------------------------------

n = 4  								# number of datapoints
rows = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
cols = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset
noise = 0.1								# strengt of noise

[C, R] = np.meshgrid(cols, rows)

print('C:', np.shape(C), 'R:', np.shape(R) )
print('col:', np.shape(cols), 'row:', np.shape(rows))

print('C:', C, 'R:', R )
print('col:', cols, 'row:', rows)

z_mesh = FrankeFunction(C,R)
z_no_mesh = FrankeFunction(cols,rows)

print('z_mesh', z_mesh)
print('z_mesh_flattend', z_mesh.flatten())
print('z_mesh_reshape', z_mesh.reshape(-1,1))
print('z_no_mesh', z_no_mesh)


X_mesh = polynomialfunction(C,R,n,5)
X_no_mesh = polynomialfunction(cols,rows,n,5)

beta_mesh = np.linalg.pinv(X_mesh.T.dot(X_mesh)).dot(X_mesh.T).dot(z_mesh)
beta_no_mesh = np.linalg.pinv(X_no_mesh.T.dot(X_no_mesh)).dot(X_no_mesh.T).dot(z_no_mesh)  

print('beta_mesh', beta_mesh)
print('beta_no_mesh', beta_no_mesh)


