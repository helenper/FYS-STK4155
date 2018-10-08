####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
from plotfunctions import *
import sys

n = 10
iterations = 10
lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
method = ['OLS', 'Ridge', 'Lasso']
seed= input('Do you want to run the program with a set seed on random numbers? If yes => true, if no => false [true/false]: ' )


# To run Franke function:
for m in method:
    print('The next method to run is: %s' %m)
    answer = input('Do you want to run for %s [y/n] ?' %m)
    if answer == 'y' or answer == 'Y': 
        file = open('results_franke_%s.txt' %m,  'w')
        for d in range(1,6):
            for elm in lambda_values: 
                mse_average, r2score_average, bias_average, var_average, beta, best_beta, mse_min, r2_min, iteration_best = runFranke(d, elm, n, iterations, seed, method=m)

                the_beta_values = []
                for i in range(best_beta.shape[0]): 
                    the_beta_values.append(np.mean(best_beta[i]))

                confidenceIntervall = betaConfidenceInterval(beta, best_beta, iteration_best)

                file.write('The results from running a degree %s polynominal with lamda = %f \n' %(d, elm))
                file.write('MSE_average:        %f \n' %mse_average )
                file.write('R2_score_average:   %f \n' %r2score_average)
                file.write('Bias_average:       %f \n' %bias_average)
                file.write('Variance_average:   %f \n' %var_average)
                file.write('Min_MSE_value:      %f \n' %mse_min)
                file.write('R2_for_Min_MSE_value:       %f \n' %r2_min)
                file.write('The_best_beta_parameters:   %s \n' %the_beta_values)
                file.write('Confidence_intervall:   %s \n' %confidenceIntervall)
                file.write('\n')

                
        file.close()


    if answer == 'n' or answer == 'N':
        print('Moving on')


# To run terrain data:
for m in method:
    print('The next method to run is: %s' %m)
    answer = input('Do you want to run for %s [y/n] ?' %m)
    if answer == 'y' or answer == 'Y': 
        file = open('results_terrain_%s.txt' %m,  'w')
        for d in range(5,6):
            for elm in lambda_values: 
                mse_average, r2score_average, bias_average, var_average, beta, best_beta, mse_min, r2_min, iteration_best = runTerrain(d, elm, n, iterations, seed, method=m)

                confidenceIntervall = betaConfidenceInterval_terrain(beta, best_beta, iteration_best)

                file.write('The results from running a degree %s polynominal with lamda = %f \n' %(d, elm))
                file.write('MSE_average:        %s \n' %mse_average )
                file.write('R2_score_average:   %s \n' %r2score_average)
                file.write('Bias_average:       %s \n' %bias_average)
                file.write('Variance_average:   %s \n' %var_average) 
                file.write('Min_MSE_value:      %s \n' %mse_min)
                file.write('R2_for_Min_MSE_value:       %s \n' %r2_min)
                file.write('The_best_beta_parameters:   %s \n' %best_beta)
                file.write('Confidence_intervall:   %s \n' %confidenceIntervall)
                file.write('\n')

                
        file.close()


    if answer == 'n' or answer == 'N':
        print('Moving on')



# For plotting: 
mse_OLS, r2_OLS, bias_OLS, var_OLS = retrive_data_from_file('results_franke_OLS.txt', 5, 6)
mse_Ridge, r2_Ridge, bias_Ridge, var_Ridge = retrive_data_from_file('results_franke_Ridge.txt', 5, 6)
mse_Lasso, r2_Lasso, bias_Lasso, var_Lasso = retrive_data_from_file('results_franke_Lasso.txt', 5, 6)
plotMSE_Ridge_OLS(mse_Ridge, mse_OLS)
plotMSE_Lasso(mse_Lasso)
plotR2_Ridge_OLS(r2_Ridge, r2_OLS)
plotR2_Lasso(r2_Lasso)

mse_OLS, r2_OLS, bias_OLS, var_OLS = retrive_data_from_file_terrain('results_terrain_OLS.txt', 5, 6)
mse_Ridge, r2_Ridge, bias_Ridge, var_Ridge = retrive_data_from_file_terrain('results_terrain_Ridge.txt', 5, 6)
mse_Lasso, r2_Lasso, bias_Lasso, var_Lasso = retrive_data_from_file_terrain('results_terrain_Lasso.txt', 5, 6)
plotOLS(mse_OLS, r2_OLS, bias_OLS, var_OLS)
plotRidge(mse_Ridge, r2_Ridge, bias_Ridge, var_Ridge)
plotLasso(mse_Lasso, r2_Lasso, bias_Lasso, var_Lasso)

















