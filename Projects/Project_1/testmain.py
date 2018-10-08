####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
from plotfunctions import *
import sys

n = 100
iterations = 100
lambda_values = [1e-4]#, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
method = ['OLS', 'Ridge', 'Lasso']
seed= input('Do you want to run the program with a set seed on random numbers? If yes => true, if no => false [true/false]: ' )


for m in method:
    print('The next method to run is: %s' %m)
    answer = input('Do you want to run for %s [y/n] ?' %m)
    if answer == 'y' or answer == 'Y': 
        file = open('results_terrain_%s_plot.txt' %m,  'w')
        for d in range(5,6):
            for elm in lambda_values: 
                mse_average, r2score_average, bias_average, var_average, beta, best_beta, mse_min, r2_min, iteration_best = runTerrain(d, elm, n, iterations, seed, method=m)
               # print(np.shape(beta))
                #print(beta[iteration_best])
                #the_beta_values = []
                #for i in range(len(best_beta)): 
                #    the_beta_values.append(np.mean(best_beta[i]))
                #print(the_beta_values)

                confidenceIntervall = betaConfidenceInterval_terrain(beta, best_beta, iteration_best)


                file.write('The results from running a degree %s polynominal with lamda = %f \n' %(d, elm))
                file.write('MSE_average:        %s \n' %mse_average )
                file.write('R2_score_average:   %s \n' %r2score_average)
                file.write('Bias_average:       %s \n' %bias_average)
                file.write('Variance_average:   %s \n' %var_average)
                #file.write('Best_beta_values:  %s \n' %best_beta) 
                file.write('Min_MSE_value:      %s \n' %mse_min)
                file.write('R2_for_Min_MSE_value:       %s \n' %r2_min)
                file.write('The_best_beta_parameters:   %s \n' %best_beta)
                file.write('Confidence_intervall:   %s \n' %confidenceIntervall)
                file.write('\n')

                
        file.close()


    if answer == 'n' or answer == 'N':
        print('Moving on')


#print('beta:', np.shape(beta))
#print('best beta', np.shape(best_beta))