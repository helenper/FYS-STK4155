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
# Make test-dataset and calculate Franke's function
#---------------------------------------------------

n = 1000    							# number of datapoints
x = np.random.uniform(0.0,1.0, n)       # create a random number for x-values in dataset
y = np.random.uniform(0.0,1.0, n)       # create a random number for y-values in dataset
noise = 0.1								# strengt of noise

# Define the Franke function from our test-dataset
z = FrankeFunction(x, y) + noise*np.random.randn(1) # z with noise



#---------------------------------------------------------------------
# Use bootstrap to define train and test data and calculate a mean 
# value for MSE and R2 for the different methods OSL, Ridge and Lasso
#---------------------------------------------------------------------

iterations = 1000		# number of times we split and save our calculations in train and test point

# Create arrays to hold different values to be taken mean over later. 
# Each arrray is a nested array, where the first index points to the degree of the polynomial
# used in that iteration. 
mse_OLS = np.zeros((5,iterations))
mse_Ridge = np.zeros((5,iterations))
mse_Lasso = np.zeros((5,iterations))
r2score_OLS = np.zeros((5,iterations))
r2score_Ridge = np.zeros((5,iterations))
r2score_Lasso = np.zeros((5,iterations))


# Parameter to be sendt into Lasso and Rigde 
alpha = 0.001
train_indices, test_indices = bootstrap(z, 0.7)




for i in range(iterations):
    
    for j in range(5):
        X = polynomialfunction(x,y,n,degree=(j+1))
        X_train = X[train_indices]; #print(X_train.shape)
        np.random.shuffle(X_train)
        X_test = X[test_indices]; #print(X_test.shape)
        z_train = z[train_indices]; #print(z_train.shape)
        z_test = z[test_indices]; #print(z_test.shape)

        mse_OLS[j][i], r2score_OLS[j][i] = OLS(X_train,z_train, X_test)

        mse_Ridge[j][i], r2score_Ridge[j][i] = ridge(x,y,z_train,X_train,X_test,alpha, write=0)

        mse_Lasso[j][i], r2score_Lasso[j][i] = lasso(X_train,z_train,X_test,alpha)


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



var=1.0/z.shape[0] *np.sum((z - np.mean(z))**2)
#print(var)