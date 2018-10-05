####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################

from functions import *
from plotfunctions import *
from imageio import imread

#-------------------------------------------------
# Define a seed to use while testing the program. 
# Comment out this to have changing random numbers. 
np.random.seed(4155)
#-------------------------------------------------


#---------------------------------------------------
# Load the terrain
#terrain1 = imread('terrainone.tif')
#x, y = imread('terrainone.tif')
terrain1 = imread('terrainone.tif')
[n,m] = terrain1.shape

    ## Find some random patches within the dataset and perform a fit

patch_size_row = 100
patch_size_col = 100

    # Define their axes
rows = np.linspace(0,1,patch_size_row)
cols = np.linspace(0,1,patch_size_col)

#[C,R] = np.meshgrid(cols,rows)

#print('hei', np.size(C), np.size(R))

#x = C.reshape(-1,1)
#y = R.reshape(-1,1)
x = rows
y = cols

#print(len(x))
#print(np.shape(x), np.shape(y))

num_data = patch_size_row*patch_size_col

    # Find the start indices of each patch

num_patches = 3

np.random.seed(4155)



row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)
# Show the terrain
#print(np.shape(terrain1))


for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
	row_end = row_start + patch_size_row
	col_end = col_start + patch_size_col

	patch = terrain1[row_start:row_end, col_start:col_end]
	z = patch
	#print(np.shape(z))
	
	train_indices, test_indices = splitdata(x, 0.7)

	x_train_orig = x[train_indices]
	x_test = x[test_indices]
	y_train_orig = y[train_indices]
	y_test = y[test_indices]

	z_train = z[train_indices]
	z_test = z[test_indices]

	#print(np.shape(x_train_orig), np.shape(y_train_orig), np.shape(z_train))

	X_train = polynomialfunction(x_train_orig, y_train_orig, len(x_train_orig), degree=5)
	X_test = polynomialfunction(x_test, y_test, len(x_test), degree=5)

	print(np.shape(X_train), np.shape(X_test))

	mse , R2, bias, var, beta = OLS(X_train, z_train, X_test, z_test)
	print('OLS:','mse',mse, 'R2',R2, 'bias',bias,'var', var)

	alpha = 0.01
	mse_R , R2_R, bias_R, var_R = ridge(X_train, z_train, X_test, z_test, alpha, write=0)
	
	print('Rigde:', 'mse',mse_R, 'R2',R2_R, 'bias',bias_R,'var', var_R)


	mse_L , R2_L, bias_L, var_L = lasso(X_train,z_train,X_test,z_test,alpha, write=0)

	print('Lasso:', 'mse',mse_L, 'R2',R2_L, 'bias',bias_L,'var', var_L)