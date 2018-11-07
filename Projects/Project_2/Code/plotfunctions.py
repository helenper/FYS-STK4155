# Plotting:
# Importing necessary packages for plotting
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import re

def retrive_data_from_file(filename, num_degree, numb_of_lambda):
    infile = open(filename, 'r')

    MSE_average = []
    R2_average = []
    Bias_average = []
    Variance_average =[]

    for lines in infile: 
        line = lines.split()
        if len(line) == 2:
            if line[0] == 'MSE_average:' :
                MSE_average.append(float(line[1]))
            
            if line[0] == 'R2_score_average:':
                R2_average.append(float(line[1]))

            if line[0] == 'Bias_average:' :
                Bias_average.append(float(line[1]))

            if line[0] == 'Variance_average:':
                Variance_average.append(float(line[1]))
            


    infile.close()

    return MSE_average, R2_average, Bias_average, Variance_average

def plotMSE_Ridge_OLS(mse, mse_OLS):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse[0:6], 'b')
    plt.plot(lambda_values, mse[6:12], 'c')
    plt.plot(lambda_values, mse[12:18], 'g')
    plt.plot(lambda_values, mse[18:24], 'r')
    plt.plot(lambda_values, mse[24:30], 'm')
    plt.plot( lambda_values, mse_OLS[0:6], 'bo')
    plt.plot(lambda_values, mse_OLS[6:12], 'co')
    plt.plot(lambda_values, mse_OLS[12:18], 'go')
    plt.plot(lambda_values, mse_OLS[18:24], 'ro')
    plt.plot(lambda_values, mse_OLS[24:30], 'mo')

    plt.title('MSE calculated by Ridge and OLS')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.semilogx()
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.show()




def plotMSE_Lasso(mse):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, mse[0:6])
    plt.plot(lambda_values, mse[6:12])
    plt.plot(lambda_values, mse[12:18])
    plt.plot(lambda_values, mse[18:24])
    plt.plot(lambda_values, mse[24:30])
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title('MSE calculated by Lasso')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.semilogx()
    plt.show()


def plotR2_Ridge_OLS(r2, r2_OLS):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, r2[0:6], 'b')
    plt.plot(lambda_values, r2[6:12], 'c')
    plt.plot(lambda_values, r2[12:18], 'g')
    plt.plot(lambda_values, r2[18:24], 'r')
    plt.plot(lambda_values, r2[24:30], 'm')
    
    plt.plot( lambda_values, r2_OLS[0:6], 'bo')
    plt.plot(lambda_values, r2_OLS[6:12], 'co')
    plt.plot(lambda_values, r2_OLS[12:18], 'go')
    plt.plot(lambda_values, r2_OLS[18:24], 'ro')
    plt.plot(lambda_values, r2_OLS[24:30], 'mo')
    
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title(r'R$^2$ score calculated by Ridge and OLS')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'R$^2$ score')
    plt.semilogx()
    plt.show()


def plotR2_Lasso(r2):
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    plt.plot(lambda_values, r2[0:6])
    plt.plot(lambda_values, r2[6:12])
    plt.plot(lambda_values, r2[12:18])
    plt.plot(lambda_values, r2[18:24])
    plt.plot(lambda_values, r2[24:30])
    plt.legend(['Deg = 1', 'Deg = 2', 'Deg = 3', 'Deg = 4', 'Deg = 5'])
    plt.title(r'R$^2$ score calculated by Lasso')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'R$^2$ score')
    plt.semilogx()
    plt.show()  

 


def retrive_data_from_file_terrain(filename, num_degree, numb_of_lambda):
    infile = open(filename, 'r')

    MSE_average = []
    R2_average = []
    Bias_average = []
    Variance_average =[]

    for lines in infile: 
        line = lines.split()
        if len(line) >= 2:
            if line[0] == 'MSE_average:' :
                line[1] =re.sub("\\[","",line[1])
                line[-1] = re.sub("\\]","",line[-1])
                for i in range(1,len(line)):
                    MSE_average.append(line[i])

            if line[0] == 'R2_score_average:':
                line[1] =re.sub("\\[","",line[1])
                line[-1] = re.sub("\\ ]","",line[-1])
                for i in range(1,len(line)):
                    R2_average.append(line[i])

            if line[0] == 'Bias_average:' :
                line[1] =re.sub("\\[ ","",line[1])
                line[-1] = re.sub("\\]","",line[-1])
                for i in range(1,len(line)):
                    Bias_average.append(line[i])

            if line[0] == 'Variance_average:':
                line[1] =re.sub("\\[","",line[1])
                line[-1] = re.sub("\\]","",line[-1])
                for i in range(1,len(line)):
                    Variance_average.append(line[i])
            
    infile.close()

    return MSE_average, R2_average, Bias_average, Variance_average

def plotOLS(mse, r2, bias, var):

    poly = [1, 2, 3, 4, 5]
    
    mse_list = [float(mse[1]), float(mse[32]), float(mse[68]), float(mse[104]), float(mse[140])]
    r2_list = [float(r2[1]), float(r2[37]), float(r2[67]), float(r2[103]), float(r2[138])]
    bias_list = [float(bias[2]), float(bias[38]), float(bias[74]), float(bias[116]), float(bias[140])]
    var_list = [float(var[1]), float(var[31]), float(var[66]), float(var[101]), float(var[146])]

    print(mse_list)
    print(r2_list)
    print(bias_list)
    print(var_list)

    plt.plot(poly,mse_list)
    plt.plot(poly, r2_list)
    plt.plot(poly, bias_list)
    plt.plot(poly, var_list)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with OLS')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()


def plotRidge(mse, r2, bias, var):
    poly = [1, 2, 3, 4, 5]

    mse_list = [float(mse[1]), float(mse[32]), float(mse[68]), float(mse[104]), float(mse[139])]
    r2_list = [float(r2[1]), float(r2[32]), float(r2[67]), float(r2[93]), float(r2[123])]
    bias_list = [float(bias[2]), float(bias[33]), float(bias[76]), float(bias[112]), float(bias[148])]
    var_list = [float(var[1]), float(var[33]), float(var[68]), float(var[93]), float(var[123])]
    print(mse_list)
    print(r2_list)
    print(bias_list)
    print(var_list)

    plt.plot(poly,mse_list)
    plt.plot(poly, r2_list)
    plt.plot(poly, bias_list)
    plt.plot(poly, var_list)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with Ridge')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()

def plotLasso(mse, r2, bias, var):
    #poly = [i+1 for i in range(len(mse))]
    poly = [1, 2, 3, 4]
    
    mse_list = [float(mse[2]), float(mse[7]), float(mse[12]), float(mse[17])]
    r2_list = [float(r2[1]), float(r2[5]), float(r2[9]), float(r2[13])]
    bias_list = [float(bias[2]), float(bias[8]), float(bias[14]), float(bias[20])]
    var_list = [float(var[1]), float(var[6]), float(var[12]), float(var[17])]
    print(mse_list)
    print(r2_list)
    print(bias_list)
    print(var_list)

    plt.plot(poly,mse_list)
    plt.plot(poly, r2_list)
    plt.plot(poly, bias_list)
    plt.plot(poly, var_list)
    plt.legend(['MSE', r'R$^2$ score', r'Bias$^2$', 'Variance'])
    plt.title('Different statistical qualitites calculated with Lasso')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Statistics')
    plt.show()



def Plot_Accuracy(acc, eta):

    xaxis = np.linspace(0,len(acc)-1, len(acc))
    plt.plot(xaxis , acc, 'bo', markersize=2, label='Training accuracy')
    plt.title("Accuracy for the neural network training on two dimensional Ising-model with eta=%f." % eta)
    plt.xlabel("Number of iterations")
    plt.ylabel("Percentage of correct predictions")
    plt.legend()
    plt.show()
