####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################

from functions import *
from plotfunctions import *
from imageio import imread
from sklearn.model_selection import train_test_split

#-------------------------------------------------
# Define a seed to use while testing the program. 
# Comment out this to have changing random numbers. 
np.random.seed(4155)
#-------------------------------------------------


#---------------------------------------------------
# Make test-dataset and calculate Franke's function
#---------------------------------------------------
"""
n = 10    								# number of datapoints
x = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset
noise = 0.1								# strengt of noise
"""
#---------------------------------------------------
# Load the terrain
#terrain1 = imread('terrainone.tif')
#x, y = imread('terrainone.tif')
terrain1 = imread('terrainone.tif')
[n,m] = terrain1.shape

    ## Find some random patches within the dataset and perform a fit

patch_size_row = 100
patch_size_col = 50

    # Define their axes
rows = np.linspace(0,1,patch_size_row)
cols = np.linspace(0,1,patch_size_col)

[C,R] = np.meshgrid(cols,rows)

#print('hei', np.size(C), np.size(R))

x = C.reshape(-1,1)
y = R.reshape(-1,1)
print(len(x))
#print(np.shape(x), np.shape(y))

num_data = patch_size_row*patch_size_col

    # Find the start indices of each patch

num_patches = 5

np.random.seed(4155)



row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)
# Show the terrain
#print(np.shape(terrain1))




#---------------------------------------------------------




# Define the Franke function from our test-dataset

#z_test = FrankeFunction(x_test, y_test) + noise*np.random.randn(len(x_test)) # z with noise



#---------------------------------------------------------------------
# Use bootstrap to define train and test data and calculate a mean 
# value for MSE and R2 for the different methods OSL, Ridge and Lasso
#---------------------------------------------------------------------

iterations = 10		# number of times we split and save our calculations in train and test point

# Create arrays to hold different values to be taken mean over later. 
# Each arrray is a nested array, where the first index points to the degree of the polynomial
# used in that iteration. 



# Parameter to be sendt into Lasso and Rigde 
alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4]

mse_OLS = np.zeros((5,iterations))
mse_Ridge = np.zeros((5,iterations))
mse_Lasso = np.zeros((5,iterations))
r2score_OLS = np.zeros((5,iterations))
r2score_Ridge = np.zeros((5,iterations))
r2score_Lasso = np.zeros((5,iterations))
bias_OLS = np.zeros((5,iterations))
bias_Ridge = np.zeros((5,iterations))
bias_Lasso = np.zeros((5,iterations))
var_OLS = np.zeros((5,iterations))
var_Ridge = np.zeros((5,iterations))
var_Lasso = np.zeros((5,iterations))
beta = np.zeros(iterations)


file = open('data_terrain.txt','w') #Change filename
beta_file = open('beta_data_terrain.txt', 'w')



for k,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
	file.write('%d    \n' %(k+1))
	row_end = row_start + patch_size_row
	col_end = col_start + patch_size_col

	patch = terrain1[row_start:row_end, col_start:col_end]
	#print(np.shape(patch))

	z = patch.reshape(-1,1)
	for j in range(5):
		X = polynomialfunction(x,y,len(x),degree=(j+1))
		X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = 0.7)
#file.write('alpha     mse_OLS_average1 \n')
		for a in alpha:
			file.write('%f   \n' %a)
			for i in range(iterations):
				#print(np.shape(X_train))
				X_train, z_train = bootstrap(X_train, z_train)
				#print(np.shape(X_train))
		#X_train, z_train = bootstrap(X_train,z_train)
		#z_train = FrankeFunction(x_train, y_train) + noise*np.random.randn(len(x_train)) # z with noise

				
			#X = polynomialfunction(x,y,len(x),4)

				

				#mse_OLS, r2_score_OLS, bias_OLS, var_OLS, beta = OLS(X_train,z_train, X_test, z_test)

	#print(mse_OLS, r2_score_OLS)
			
	        #X_test = polynomialfunction(x_test,y_test,len(x_test),degree=(j+1))
	        

				mse_OLS[j][i], r2score_OLS[j][i], bias_OLS[j][i], var_OLS[j][i], beta = OLS(X_train,z_train, X_test, z_test)
		        #print(np.shape(beta))

		        #betaConfidenceInterval(beta, beta_file)

				mse_Ridge[j][i], r2score_Ridge[j][i], bias_Ridge[j][i], var_Ridge[j][i] = ridge(X_train,z_train,X_test,z_test,a, write=0)

				mse_Lasso[j][i], r2score_Lasso[j][i], bias_Lasso[j][i], var_Lasso[j][i] = lasso(X_train,z_train,X_test,z_test,a, write=0)

		    #print(mse_OLS[i], r2score_OLS[i])

				mse_OLS_average1 = np.mean(mse_OLS[0])	
				mse_OLS_average2 = np.mean(mse_OLS[1]) 
				mse_OLS_average3 = np.mean(mse_OLS[2]) 
				mse_OLS_average4 = np.mean(mse_OLS[3]) 
				mse_OLS_average5 = np.mean(mse_OLS[4])

				mse_Ridge_average1 = np.mean(mse_Ridge[0]) 
				mse_Ridge_average2 = np.mean(mse_Ridge[1])
				mse_Ridge_average3 = np.mean(mse_Ridge[2])
				mse_Ridge_average4 = np.mean(mse_Ridge[3])
				mse_Ridge_average5 = np.mean(mse_Ridge[4])

				mse_Lasso_average1 = np.mean(mse_Lasso[0])
				mse_Lasso_average2 = np.mean(mse_Lasso[1])
				mse_Lasso_average3 = np.mean(mse_Lasso[2])
				mse_Lasso_average4 = np.mean(mse_Lasso[3])
				mse_Lasso_average5 = np.mean(mse_Lasso[4])

				r2score_OLS_average1 = np.mean(r2score_OLS[0])
				r2score_OLS_average2 = np.mean(r2score_OLS[1])
				r2score_OLS_average3 = np.mean(r2score_OLS[2])
				r2score_OLS_average4 = np.mean(r2score_OLS[3])
				r2score_OLS_average5 = np.mean(r2score_OLS[4])

				r2score_Ridge_average1 = np.mean(r2score_Ridge[0])
				r2score_Ridge_average2 = np.mean(r2score_Ridge[1])
				r2score_Ridge_average3 = np.mean(r2score_Ridge[2])
				r2score_Ridge_average4 = np.mean(r2score_Ridge[3])
				r2score_Ridge_average5 = np.mean(r2score_Ridge[4])

				r2score_Lasso_average1 = np.mean(r2score_Lasso[0])
				r2score_Lasso_average2 = np.mean(r2score_Lasso[1])
				r2score_Lasso_average3 = np.mean(r2score_Lasso[2])
				r2score_Lasso_average4 = np.mean(r2score_Lasso[3])
				r2score_Lasso_average5 = np.mean(r2score_Lasso[4])

				bias_OLS_average1 = np.mean(bias_OLS[0])	
				bias_OLS_average2 = np.mean(bias_OLS[1]) 
				bias_OLS_average3 = np.mean(bias_OLS[2]) 
				bias_OLS_average4 = np.mean(bias_OLS[3]) 
				bias_OLS_average5 = np.mean(bias_OLS[4])

				bias_Ridge_average1 = np.mean(bias_Ridge[0]) 
				bias_Ridge_average2 = np.mean(bias_Ridge[1])
				bias_Ridge_average3 = np.mean(bias_Ridge[2])
				bias_Ridge_average4 = np.mean(bias_Ridge[3])
				bias_Ridge_average5 = np.mean(bias_Ridge[4])

				bias_Lasso_average1 = np.mean(bias_Lasso[0])
				bias_Lasso_average2 = np.mean(bias_Lasso[1])
				bias_Lasso_average3 = np.mean(bias_Lasso[2])
				bias_Lasso_average4 = np.mean(bias_Lasso[3])
				bias_Lasso_average5 = np.mean(bias_Lasso[4])

				var_OLS_average1 = np.mean(var_OLS[0])	
				var_OLS_average2 = np.mean(var_OLS[1]) 
				var_OLS_average3 = np.mean(var_OLS[2]) 
				var_OLS_average4 = np.mean(var_OLS[3]) 
				var_OLS_average5 = np.mean(var_OLS[4])

				var_Ridge_average1 = np.mean(var_Ridge[0]) 
				var_Ridge_average2 = np.mean(var_Ridge[1])
				var_Ridge_average3 = np.mean(var_Ridge[2])
				var_Ridge_average4 = np.mean(var_Ridge[3])
				var_Ridge_average5 = np.mean(var_Ridge[4])

				var_Lasso_average1 = np.mean(var_Lasso[0])
				var_Lasso_average2 = np.mean(var_Lasso[1])
				var_Lasso_average3 = np.mean(var_Lasso[2])
				var_Lasso_average4 = np.mean(var_Lasso[3])
				var_Lasso_average5 = np.mean(var_Lasso[4])

				file.write('%f   %f     %f      %f    %f    ' %(mse_OLS_average1, mse_OLS_average2, mse_OLS_average3, mse_OLS_average4, mse_OLS_average5))
				file.write('%f   %f     %f      %f    %f    ' %(mse_Ridge_average1, mse_Ridge_average2, mse_Ridge_average3, mse_Ridge_average4, mse_Ridge_average5))
				file.write('%f   %f     %f      %f    %f    \n' %(mse_Lasso_average1, mse_Lasso_average2, mse_Lasso_average3, mse_Lasso_average4, mse_Lasso_average5))

				file.write('%f   %f     %f      %f    %f    ' %(r2score_OLS_average1, r2score_OLS_average2, r2score_OLS_average3, r2score_OLS_average4, r2score_OLS_average5))
				file.write('%f   %f     %f      %f    %f    ' %(r2score_Ridge_average1, r2score_Ridge_average2, r2score_Ridge_average3, r2score_Ridge_average4, r2score_Ridge_average5))
				file.write('%f   %f     %f      %f    %f    \n' %(r2score_Lasso_average1, r2score_Lasso_average2, r2score_Lasso_average3, r2score_Lasso_average4, r2score_Lasso_average5))

				file.write('%f   %f     %f      %f    %f    ' %(bias_OLS_average1, bias_OLS_average2, bias_OLS_average3, bias_OLS_average4, bias_OLS_average5))
				file.write('%f   %f     %f      %f    %f    ' %(bias_Ridge_average1, bias_Ridge_average2, bias_Ridge_average3, bias_Ridge_average4, bias_Ridge_average5))
				file.write('%f   %f     %f      %f    %f    \n' %(bias_Lasso_average1, bias_Lasso_average2, bias_Lasso_average3, bias_Lasso_average4, bias_Lasso_average5))

				file.write('%f   %f     %f      %f    %f    ' %(var_OLS_average1, var_OLS_average2, var_OLS_average3, var_OLS_average4, var_OLS_average5))
				file.write('%f   %f     %f      %f    %f    ' %(var_Ridge_average1, var_Ridge_average2, var_Ridge_average3, var_Ridge_average4, var_Ridge_average5))
				file.write('%f   %f     %f      %f    %f    \n' %(var_Lasso_average1, var_Lasso_average2, var_Lasso_average3, var_Lasso_average4, var_Lasso_average5))



file.close()
beta_file.close()

"""
#print(np.size(interval))
#print('hei')
#betaConfidenceInterval(beta)
#plotMSE()
"""
"""
print("Datasize = ", n, "\n")
print("alpha = ", alpha, "\n")
print("bootstrap-iterations = ", iterations, "\n")


print("The average mean sqared error for the different polynomial powers: \n")
print("OLS: ")
print("1. order: ", mse_OLS_average1, "2. order: ", mse_OLS_average2, "3. order: ", mse_OLS_average3, "4. order: ", mse_OLS_average4, "5. order: ", mse_OLS_average5, "\n")
print("Ridge: ")
print("1. order: ", mse_Ridge_average1, "2. order: ", mse_Ridge_average2, "3. order: ", mse_Ridge_average3, "4. order: ", mse_Ridge_average4, "5. order: ", mse_Ridge_average5, "\n")
print("Lasso: ")
print("1. order: ", mse_Lasso_average1, "2. order: ", mse_Lasso_average2, "3. order: ", mse_Lasso_average3, "4. order: ", mse_Lasso_average4, "5. order: ", mse_Lasso_average5, "\n")

print("")
print("----------------------------------------------------------------------------------------------------------")
print("")

print("The average R2 score for the different polynomial powers: \n")
print("OLS: ")
print("1. order: ", r2score_OLS_average1, "2. order: ", r2score_OLS_average2, "3. order: ", r2score_OLS_average3, "4. order: ", r2score_OLS_average4, "5. order: ", r2score_OLS_average5, "\n")
print("Ridge: ")
print("1. order: ", r2score_Ridge_average1, "2. order: ", r2score_Ridge_average2, "3. order: ", r2score_Ridge_average3, "4. order: ", r2score_Ridge_average4, "5. order: ", r2score_Ridge_average5, "\n")
print("Lasso: ")
print("1. order: ", r2score_Lasso_average1, "2. order: ", r2score_Lasso_average2, "3. order: ", r2score_Lasso_average3, "4. order: ", r2score_Lasso_average4, "5. order: ", r2score_Lasso_average5, "\n")
"""


#var=1.0/z.shape[0] *np.sum((z - np.mean(z))**2)
#print(var)
"""
	x_patch = [i + 1 for i in range(patch_size_row)] + row_start
	y_patch = [i + 1 for i in range(patch_size_col)] + col_start
	

	patch = terrain1[row_start:row_end, col_start:col_end]
	z = patch.reshape(-1,1)
	#print(row_start, row_end)
	print(np.shape(patch), np.shape(z))
	#print(len(patch))
	N = terrain1[row_start:row_end]
	train_indices, test_indices = splitdata(z, 0.7)
	#x_train_indices, x_test_indices = splitdata(x_patch, 0.7)
	#y_train_indices, y_test_indices = splitdata(y_patch, 0.7)
	#print(np.shape(train_indices), np.shape(test_indices))
	#print(len(train_indices))
	#from sklearn.model_selection import train_test_split
	#X = polynomialfunction(x, y, len(x), degree=5)
	#X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
	
	x_train_orig = x[train_indices]
	x_test = x[test_indices]
	y_train_orig = y[train_indices]
	y_test = y[test_indices]

	z_train = z[train_indices]
	z_test = z[test_indices]
	
	#z_train = patch_train.reshape(-1,1)
	#z_test = patch_test.reshape(-1,1)
	#print(np.shape(x_train_orig), np.shape(y_train_orig), np.shape(z_train), num_data)
	
	X_train = polynomialfunction(x_train_orig, y_train_orig, len(x_train_orig), degree=2)
	X_test = polynomialfunction(x_test, y_test, len(x_test), degree=2)

	#print(np.shape(X_train), np.shape(z_train), np.shape(X_test), np.shape(z_test))

	mse , R2, bias, var, beta = OLS(X_train, z_train, X_test, z_test)

	print('mse',mse, 'R2',R2, 'bias',bias,'var', var)

	alpha = 0.001
	mse_R , R2_R, bias_R, var_R = ridge(X_train, z_train, X_test, z_test, alpha, write=0)
	
	print('Rigde:', 'mse',mse_R, 'R2',R2_R, 'bias',bias_R,'var', var_R)


	mse_L , R2_L, bias_L, var_L = lasso(X_train,z_train,X_test,z_test,alpha, write=0)

	print('Lasso:', 'mse',mse_L, 'R2',R2_L, 'bias',bias_L,'var', var_L)
"""
