# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:11:09 2016

@author: josh
"""

import numpy as np
from sklearn import svm, grid_search, datasets
from sklearn.svm import SVR
from sklearn import preprocessing
from multiprocessing import cpu_count

numTest = 2000
O1rawData = np.loadtxt(open("O1_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )
H2rawData = np.loadtxt(open("H2_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )
H3rawData = np.loadtxt(open("H3_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )

# Selected the valid moments, which change appreciably and excluded non-descriptive ones.
O1moments = np.column_stack( (O1rawData[:,3], O1rawData[:,5:8], O1rawData[:,10:12], 
                              O1rawData[:,13:15], O1rawData[:,17:20], 
O1rawData[:,22:24], O1rawData[:,26:28]) )

H2moments = np.column_stack( (H2rawData[:,3], H2rawData[:,5:8], H2rawData[:,10:12], 
                              H2rawData[:,13:15], H2rawData[:,17:20], 
H2rawData[:,22:24], H2rawData[:,26:28]) )

H3moments = np.column_stack( (H3rawData[:,3], H3rawData[:,5:8], H3rawData[:,10:12], 
                              H3rawData[:,13:15], H3rawData[:,17:20], 
H3rawData[:,22:24], H3rawData[:,26:28]) )

geometry = O1rawData[:,:3]

std_O1moments = O1moments
std_geometry = geometry
robust_O1moments = O1moments
robust_geometry = geometry

std_O1momentsTrans = preprocessing.StandardScaler().fit(std_O1moments) 	# To fit standardisation transform
std_O1moments = std_O1momentsTrans.transform(std_O1moments)			# To transform

std_geometryTrans = preprocessing.StandardScaler().fit(std_geometry) 	# To fit standardisation transform
std_geometry = std_geometryTrans.transform(std_geometry)			# To transform

# scale the data using robust scaler
robust_O1momentsTrans = preprocessing.RobustScaler().fit(robust_O1moments) 	# To fit standardisation transform
robust_O1moments = robust_O1momentsTrans.transform(robust_O1moments)		# To transform

robust_geometryTrans = preprocessing.RobustScaler().fit(robust_geometry) 	# To fit standardisation transform
robust_geometry = robust_geometryTrans.transform(robust_geometry)			# To transform


robust_O1_q00_model = SVR()
std_O1_q00_model = SVR()

#===============================================================================================================================
# Perhaps we can simplify life and just have both scalers run, and simply calculate the monopole: Agreed this is fine comb stuff
#-------------------------------------------------------------------------------------------------------------------------------
#userChoice = input('Which scaler would you like to use (standard/robust/both)? ')
# Error handling
#if (userChoice.lower() != 'both')| (userChoice.lower() != 'standard')| (userChoice.lower() != 'robust') :
#	while not((userChoice.lower() == 'both') | (userChoice.lower() == 'standard') | (userChoice.lower() == 'robust')):
#		userChoice = input('Sorry what was that (standard/robust/both)? ')
#===============================================================================================================================

std_parameters = {}
robust_parameters = {}


choiceParams = input('Would you like to run a parameter search, or accept the already calculated parameters? (y/n) \n >\t ')

# Error handling
if (choiceParams.lower() != 'y') | (choiceParams.lower() != 'n') :
	while not((choiceParams.lower() == 'y') | (choiceParams.lower() == 'n')):
		choiceParams = input('Sorry what was that ? ')

if choiceParams == "y":
#    std_parameters = {'kernel':['rbf'], 'C':np.arange(155,158,0.5), 
#    'gamma':np.arange(0.475,0.485,0.001), 'epsilon':[0.001] } 
    
#    robust_parameters = {'kernel':['rbf'], 'C':np.arange(500,1000,100), 
#    'gamma':np.arange(0.18,0.3,0.01), 'epsilon':[0.001]  }
	
    std_parameters = {'kernel':['rbf'], 'C':np.arange(156,158,1), 
    'gamma':np.arange(0.475,0.485,0.01), 'epsilon':[0.001] } 
    
    robust_parameters = {'kernel':['rbf'], 'C':np.arange(800,1000,100), 
    'gamma':np.arange(0.19,0.21,0.01), 'epsilon':[0.001]  }
    
    # Unscaled 
    #====================
    #O1_q00_trainIn = geometry[:-1*numTest,:]
    #O1_q00_trainOut = O1moments[:-1*numTest, 0]
    #O1_q00_testIn = geometry[-1*numTest:,:] 
    
    # Standard scaled
    #====================
    std_O1_q00_trainIn = std_geometry[:-1*numTest,:]
    std_O1_q00_trainOut = std_O1moments[:-1*numTest, 0]
    std_O1_q00_testIn = std_geometry[-1*numTest:,:] 
    
    std_paramSearch = grid_search.GridSearchCV(std_O1_q00_model, std_parameters, n_jobs=cpu_count())
    std_paramSearch.fit(std_O1_q00_trainIn, std_O1_q00_trainOut)
    std_paramSearchOutput = std_paramSearch.best_params_
    print( "Best C, \"penalty for error\" was \t\t {}".format(std_paramSearchOutput['C'] ))
    print( "Best gamma, \"size of the RBF\'s guassian\" \t was {}".format(std_paramSearchOutput['gamma'] ))
    print( "Best epsilon, \"zero punishment envelope\" \t was {}".format(std_paramSearchOutput['epsilon'] ))
    print( "The best R2 score was {}".format(std_paramSearch.best_score_))
    
    
    # Robust scaled
    #====================
    robust_O1_q00_trainIn = robust_geometry[:-1*numTest,:]
    robust_O1_q00_trainOut = robust_O1moments[:-1*numTest, 0]
    robust_O1_q00_testIn = robust_geometry[-1*numTest:,:] 
    
    robust_paramSearch = grid_search.GridSearchCV(robust_O1_q00_model, robust_parameters, n_jobs=cpu_count())
    robust_paramSearch.fit(robust_O1_q00_trainIn, robust_O1_q00_trainOut)
    robust_paramSearchOutput = robust_paramSearch.best_params_
    print( "Best C, \"penalty for error\" was \t\t {}".format(robust_paramSearchOutput['C'] ))
    print( "Best gamma, \"size of the RBF\'s guassian\" \t was {}".format(robust_paramSearchOutput['gamma'] ))
    print( "Best epsilon, \"zero punishment envelope\" \t was {}".format(robust_paramSearchOutput['epsilon'] ))
    print( "The best R2 score was {}".format(robust_paramSearch.best_score_))



if choiceParams == "n":
    std_paramSearchOutput = {'kernel':'rbf', 'C':108, 
    'gamma':0.13, 'epsilon':0.001 } 
    
    robust_paramSearchOutput = {'kernel':'rbf', 'C':900, 
    'gamma':0.2, 'epsilon':0.001  }


# Calculate the oxygen's monopole using robust scaler

robust_O1_q00_model = SVR( kernel = robust_paramSearchOutput['kernel'], 
C = robust_paramSearchOutput['C'], gamma = robust_paramSearchOutput['gamma'], epsilon = robust_paramSearchOutput['epsilon'] )

robust_O1_q00_trainIn = robust_geometry[:-1*numTest,:]
robust_O1_q00_trainOut = robust_O1moments[:-1*numTest, 0]
robust_O1_q00_testIn = robust_geometry[-1*numTest:,:] 
robust_O1_q00_model.fit(robust_O1_q00_trainIn, robust_O1_q00_trainOut)
robust_O1_q00_predicted = robust_O1_q00_model.predict(robust_O1_q00_testIn)

# scale the data back using the inverse_transform
robust_dummy = np.zeros((numTest, robust_O1moments.shape[1]))
robust_dummy[:,0] = robust_O1_q00_predicted

robust_TransformPredOxygen = robust_O1momentsTrans.inverse_transform(robust_dummy)

import matplotlib.pyplot as plt
import seaborn
def printError( newPred, truth, title ):
    #=================================================================================
    # A log-linear histogram of the newPrediction model and oldPrediction model's error
    # 
    #   Usage:     printError( O1_q11c_predicted, O1_q11c_predictedOld, O1moments[-1*numTest:,1], 'O1_q11c' )
    #---------------------------------------------------------------------------------
    # calculate distribution of error in the new and older model
    #       Compare the newPrediction and oldPrection agaisnt the truth
    newPred_ErrorDist = np.abs( np.abs( newPred ) - np.abs( truth ) )  
    MIN, MAX = .0000001, 0.001
    plt.figure()
    
    n, bins, patches = plt.hist( newPred_ErrorDist, 
                                bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
    normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label=title )
    
    plt.gca().set_xscale("log")
    plt.xlabel( 'Magnitude of '+ title + ' error' )
    plt.ylabel( 'Cumerlative Number Fraction' )
    plt.title('Cumulative '+ title+ ' Error Distribution')
    plt.grid(True)
    plt.ylim(0, 1.065)
    plt.legend(loc='upper left', shadow=True, fontsize='large')
    plt.savefig('cumulative_'+ title+ '_error.png',dpi=600)
    plt.show()
    
    newPred_avError = np.mean( newPred_ErrorDist )
    print("The average absolute error is ", newPred_avError )

printError( robust_TransformPredOxygen[:,0], O1moments[-1*numTest:, 0], "oxygen monopole")



std_O1_q00_model = SVR( kernel = std_paramSearchOutput['kernel'], 
C = std_paramSearchOutput['C'], gamma = std_paramSearchOutput['gamma'], epsilon = std_paramSearchOutput['epsilon'] )

std_O1_q00_trainIn = std_geometry[:-1*numTest,:]
std_O1_q00_trainOut = std_O1moments[:-1*numTest, 0]
std_O1_q00_testIn = std_geometry[-1*numTest:,:] 
std_O1_q00_model.fit(std_O1_q00_trainIn, std_O1_q00_trainOut)
std_O1_q00_predicted = std_O1_q00_model.predict(std_O1_q00_testIn)

# scale the data back using the inverse_transform
std_dummy = np.zeros((numTest, std_O1moments.shape[1]))
std_dummy[:,0] = std_O1_q00_predicted

std_TransformPredOxygen = std_O1momentsTrans.inverse_transform(std_dummy)


printError( std_TransformPredOxygen[:,0], O1moments[-1*numTest:, 0], "oxygen monopole")


