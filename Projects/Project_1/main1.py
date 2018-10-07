####################
# Project 1 - main program
# FYS-STK 3155/4155
# Fall 2018 
####################


from functions import *
from plotfunctions import *
import sys

n = 10
iterations = 5
lambda_values = [1e-4]#,, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
method = ['OLS', 'Rigde', 'Lasso']
seed= input('Do you want to run the program with a set seed on random numbers? If yes => true, if no => false [true/false]: ' )


for m in method:
	#if m == 'OLS': 
	print('The next method to run is: %s' %m)
	answer = input('Do you want to run for %s [y/n] ?' %m)
	if answer == 'y' or answer == 'Y': 
		file = open('results_franke_%s.txt' %m,  'w')
		for d in range(5,6):
			for elm in lambda_values: 
				mse_average, r2score_average, bias_average, var_average, beta, best_beta, mse_min, r2_min, iteration_best = runFranke(d, elm, n, iterations,seed, method=m)
				
				confidenceIntervall = betaConfidenceInterval(beta, best_beta, iteration_best)
				print(np.mean(best_beta[iteration_best][0]), np.mean(best_beta[iteration_best][1]), np.mean(best_beta[iteration_best][2]))

				file.write('The results from running a degree %s polynominal with lamda = %f \n' %(d, elm))
				file.write('MSE_average: 		%f \n' %mse_average )
				file.write('R2_score_average:	%f \n' %r2score_average)
				file.write('Bias_average: 		%f \n' %bias_average)
				file.write('Variance_average: 	%f \n' %var_average)
				#file.write('Best_beta_values: 	%s \n' %best_beta) 
				file.write('Min_MSE_value: 		%f \n' %mse_min)
				file.write('R2_for_Min_MSE_value: 		%f \n' %r2_min)
				fild.write('The_best_beta_parameters: 	%f \n' %best_beta[iteration_best])
				file.write('Confidence_intervall: 	%f \n' %confidenceIntervall)
				file.write('\n')

				
		file.close()

	if answer == 'n' or answer == 'N':
		print('Moving on')


#print('beta:', np.shape(beta))
#print('best beta', np.shape(best_beta))