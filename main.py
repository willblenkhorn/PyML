# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:31:55 2016
Heirachical self consistent multipole approximation.
#----- Model Design ------------#
---------------------------------
1.) Estimate monopole from geometry
2.) Estimate dipole components from monopole and geometry
3.) Estimate quadrupole from from previous moments and geometry
4.) Estimate octopole from previous moments and geometry
#----- Usage ------------------#
Estimate monopole from gemoetry
Insert geometry and the results for monopole prediction into the dipole prediction
Insert geometry and the previous results... etc
    To start a document sharing server:
        infinoted --create-key --create-certificate -k key.pem  -c cert.pem
        infinoted -k key.pem  -c cert.pem
@author: josh + will
29.08 and 29.09 seconds to execute in python3 -- several updates are pending will retry

================================================================================
 multipole names 			(without excluded multipoles) 
--------------------------------------------------------------------------------
     Q[0,0] monopole, Q[1,x] dipole, Q[2,x] quadrupole, Q[3,x] octopole, ...

 multipole symbols 		(without excluded multipoles)
     Q[0,0], Q[1,1,c], Q[1,1,s], Q[2,0], Q[2,2,c], Q[2,2,s], Q[3,1,c], Q[3,1,s], 
     Q[3,3,c], Q[3,3,s], Q[4,0], Q[4,2,c], Q[4,2,s], Q[4,4,c], Q[4,4,s]
================================================================================
"""
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns # nice theme <<< OPTIONAL >>
from multiprocessing import Pool, freeze_support, cpu_count
from sklearn import preprocessing


def debugPrintFns():
    #===================================================================================
    # graph to show conservation of monopole moment is preserved in a good model
    #-----------------------------------------------------------------------------------
    sumHX_q00_predictedOld =  q00_result[2] + q00_result[4]     # These are the predicted sums (geom)
    sumHX_q00_predicted = q00_result[1] + q00_result[3]         # These are the predicted sums (geom + moments)
    sumHXmoments = H2moments + H3moments                        # These are the "true" calculated values
    plt.scatter(sumHX_q00_predictedOld, q00_result[0], c='g', s=8, label='Predicted from geometry') 
    plt.scatter( sumHXmoments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='True values')
    plt.scatter( sumHX_q00_predicted, q00_result[0], c='b', s=8, label='Predicted from monopole and geometry')
    
    # NB the prediction of oxygen is from geometry in both cases
    plt.xlabel( 'Sum Hydrogen 2 & 3 q00' )
    
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. Sum Hydrogen\'s q00 in different models' )
    plt.legend()
    plt.show()
    
    #===================================================================================
    # graph to show O1 vs H2 monopole moment is predicted well by model
    #-----------------------------------------------------------------------------------
    plt.scatter( H2moments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
    plt.scatter(q00_result[1], q00_result[0], c='b', s=8, label='predicted') # H2_new and O1 predicted values
    plt.xlabel( 'Hydrogen 2 q00' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. Hydrogen 2 q00' )
    plt.legend()
    plt.show()
    #===================================================================================
    # graph to show O1 vs H3 monopole moment is predicted well by model
    #-----------------------------------------------------------------------------------
    plt.scatter( H3moments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
    plt.scatter(q00_result[3], q00_result[0], c='b', s=8, label='predicted') # H3_new and O1_q00 predicted values
    plt.xlabel( 'Hydrogen 3 q00' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. Hydrogen 3 q00' )
    plt.legend()
    plt.show()
    #===================================================================================
    # graph to show O1 monopole moment vs bondlen1 is predicted well by model
    #-----------------------------------------------------------------------------------
    plt.scatter( geometry[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
    plt.scatter( geometry[-1*numTest:,0], q00_result[0], c='b', s=8, label='predicted') # Geometry and O1_q00
    plt.xlabel( 'bond length1' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. bond length1' )
    plt.legend()
    plt.show()
    #===================================================================================

#if __name__ == '__main__':

freeze_support() # Windows specific, multithreaded crash handling
numTest = 2000 # number of test data points out of the total input data of 3997 points  
#===============================================================================
# TODO: Create input data processesing function to select columns in input 
#       which have a mean and stdev <1E-5 (or programmable threshold)
#===============================================================================

# input data for each atom
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

# Apply standard scaler to input data
O1momentsTrans = preprocessing.StandardScaler().fit(O1moments) 	# To fit standardisation transform
O1moments = O1momentsTrans.transform(O1moments)			# To transform

H2momentsTrans = preprocessing.StandardScaler().fit(H2moments) 	# To fit standardisation transform
H2moments = O1momentsTrans.transform(H2moments)			# To transform

H3momentsTrans = preprocessing.StandardScaler().fit(H3moments) 	# To fit standardisation transform
H3moments = O1momentsTrans.transform(H3moments)			# To transform

geometryTrans = preprocessing.StandardScaler().fit(geometry) 	# To fit standardisation transform
geometry = geometryTrans.transform(geometry)			    # To transform


#===============================================================================
#     TODO: 
#-> fine grained paramerisation of the C, gamma and epsilon terms in a series of range based for loops looking at the error for a given parameter.
#===============================================================================

def monopoleModel(atom, newOrOld):
#   Purpose: Calculates the monopole value
#   Usage: Accepts atom type and if new or old model
    model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 )    
    if( newOrOld == "new" ):
        if(atom == "H2"):
            trainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
            trainOut = H2moments[:-1*numTest, 0]
            testIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted )) 
        if(atom == "H3"):
            trainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
            trainOut = H3moments[:-1*numTest, 0]
            testIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted ))
        model.fit(trainIn, trainOut)
        return model.predict(testIn)
            
    if(newOrOld=="old"):
        trainIn = geometry[:-1*numTest,:]
        testIn = geometry[-1*numTest:,:]
        if(atom == "H2"):
            trainOut = H2moments[:-1*numTest, 0]
        if(atom == "H3"):
            trainOut = H3moments[:-1*numTest, 0]  
        model.fit(trainIn, trainOut)
        return model.predict(testIn)
        
# Calculate the oxygen's monopole before starting threaded section
O1_q00_model = SVR( kernel='rbf', C=108, gamma=0.13, epsilon =0.001 ) 
O1_q00_trainIn = geometry[:-1*numTest,:]
O1_q00_trainOut = O1moments[:-1*numTest, 0]
O1_q00_testIn = geometry[-1*numTest:,:] 
O1_q00_model.fit(O1_q00_trainIn, O1_q00_trainOut)
O1_q00_predicted = O1_q00_model.predict(O1_q00_testIn)
        
#   Input arguments for multi threading "starmap" routine           
q00_inputArgs = [("H2","new"),("H2","old"),("H3","new"),("H3","old")]
q00_result = [np.empty(numTest), np.empty(numTest), np.empty(numTest),
          np.empty(numTest)]        
          
#   Multithreaded "starmap" section, each fn call passes two args
with Pool(cpu_count()) as workPool:
    q00_result = workPool.starmap(monopoleModel, q00_inputArgs)
    workPool.close()
    workPool.join()
# Result now contains, in order:  O1_old, H2_new, H2_old, H3_new, H3_old
q00_result.insert(0,O1_q00_predicted)



#======================================================================
# scale the data back using the inverse_transform
# outputTemplate matches the shape of the input moments matrix
# the moment of interest is put in the correct column in this zero'd matrix.
H2outputTemplateNew = np.zeros((numTest, H2moments.shape[1]))
H2outputTemplateOld = np.zeros((numTest, H2moments.shape[1]))
H2outputTemplateNew[:,0] = q00_result[1]
H2outputTemplateOld[:,0] = q00_result[2]

transformPredH2New = H2momentsTrans.inverse_transform(H2outputTemplateNew)
transformPredH2Old = H2momentsTrans.inverse_transform(H2outputTemplateOld)
transformTrueH2 = H2momentsTrans.inverse_transform(H2moments)


H3outputTemplateNew = np.zeros((numTest, H3moments.shape[1]))
H3outputTemplateOld = np.zeros((numTest, H3moments.shape[1]))
H3outputTemplateNew[:,0] = q00_result[3]
H3outputTemplateOld[:,0] = q00_result[4]

transformPredH3New = H2momentsTrans.inverse_transform(H3outputTemplateNew)
transformPredH3Old = H2momentsTrans.inverse_transform(H3outputTemplateOld)
transformTrueH3 = H2momentsTrans.inverse_transform(H3moments)
#======================================================================


def printError( newPred, oldPred, truth, title ):
    #=================================================================================
    # A log-linear histogram of the newPrediction model and oldPrediction model's error
    # 
    #   Usage:     printError( O1_q11c_predicted, O1_q11c_predictedOld, O1moments[-1*numTest:,1], 'O1_q11c' )
    #----------------------------------------------------------------%prun -l-----------------
    # calculate distribution of error in the new and older model
    #       Compare the newPrediction and oldPrection agaisnt the truth

    newPred_ErrorDist = np.abs( np.abs( newPred ) - np.abs( truth ) ) 
    oldPred_ErrorDist = np.abs( np.abs( oldPred ) - np.abs( truth ) ) 
    MIN, MAX = .0000001, 0.001 ## NEW
    plt.figure()
    
    n, bins, patches = plt.hist( newPred_ErrorDist, 
                                bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
    normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label='Moments and geometry' )
    
    n, bins, patches = plt.hist( oldPred_ErrorDist, 
                                bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
    normed=1,histtype='step', cumulative=True,color='b', linewidth=2, label='Geometry only' )
    
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
    oldPred_avError = np.mean( oldPred_ErrorDist )
    percImproved = ( oldPred_avError - newPred_avError ) / newPred_avError * 100
    print("The average improvement from including other moments is ", percImproved, "%" )

printError( transformPredH2New[:,0], transformPredH2Old[:,0], transformTrueH2[-1*numTest:,0], "Standard scaled new vs old model" )





## This function calculates all moments, the SVR kernel parameters for each moment would first need to be optimised


#def calcAllMoments(atom, newOrOld, momNum):
##   Purpose: Calculates the the moment value using the atom's monopole and the geometry as reference
##   Usage: Accepts atom type and if new or old model and the momentNumber
##          Moment number is used to reference the AtomMoments array inside the function
##   NOTE: This code will break when you change the number of atoms from 3, it's a hack
#    model = SVR( kernel='rbf', C=108, gamma=0.13, epsilon =0.001 )    
#    if( newOrOld == "new" ):
#        if(atom == "O1"):
#            trainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
#            trainOut = O1moments[:-1*numTest, momNum]
#            testIn = np.column_stack(( geometry[-1*numTest:,:], q00_result[0] ))
#        if(atom == "H2"):
#            trainIn = np.column_stack(( geometry[:-1*numTest,:], H2moments[:-1*numTest, 0] ))
#            trainOut = H2moments[:-1*numTest, momNum]
#            testIn = np.column_stack(( geometry[-1*numTest:,:], q00_result[1] )) 
#        if(atom == "H3"):
#            trainIn = np.column_stack(( geometry[:-1*numTest,:], H3moments[:-1*numTest, 0] ))
#            trainOut = H3moments[:-1*numTest, momNum]
#            testIn = np.column_stack(( geometry[-1*numTest:,:], q00_result[3] ))
#        model.fit(trainIn, trainOut)
#        return model.predict(testIn)
#
#    if(newOrOld=="old"):
#        trainIn = geometry[:-1*numTest,:]
#        testIn = geometry[-1*numTest:,:]
#        if(atom == "O1"):from sklearn import preprocessing
#            trainOut = O1moments[:-1*numTest, momNum]
#        if(atom == "H2"):
#            trainOut = H2moments[:-1*numTest, momNum]
#        if(atom == "H3"):
#            trainOut = H3moments[:-1*numTest, momNum]  
#        model.fit(trainIn, trainOut)
#        return model.predict(testIn)
#
##    calcAllMoments("O1", "new", 4)
#
#numAtoms = 3
#OldModel = True
## Shape returns (rows, cols), selecting npArray.shape[1] .: gives cols
#numMoments = O1moments.shape[1]  # since we caculated monopole as a special case iterate from 1
#
## Create 2D matrix of input arguments, with a tuple or arguments in each 
## First three nested lists are created and edited and the final list is then converted to a tuple
#calcAllMoments_inputArgs = [[["O1","old",j] for i in range(numAtoms*2)] for j in range(1,numMoments)]
#
## Reassign every second column to the 'new' model
#for i in range(numMoments-1): # rows for atoms
#    for j in range( 0, numAtoms*2, 2 ): # columns for multipole moments
#        calcAllMoments_inputArgs[i][j][1] = 'new'
#
#
## Reassign correct atoms to H2 atoms, stride 3
#for i in range(numMoments-1): # rows for atom's moments
#    for j in range( 2, numAtoms*2-2 ): # columns for atoms
#        calcAllMoments_inputArgs[i][j][0] = 'H2'
#
#
## Reassign correct atoms to H3 atoms, stride 3
#for i in range(numMoments-1): # rows for atom's moments
#    for j in range( 4, numAtoms*2 ): # columns for atoms
#        calcAllMoments_inputArgs[i][j][0] = 'H3'
#
#
## Reassign every element to a tuple
#for i in range(numMoments-1): # rows for atoms
#    for j in range( numAtoms*2 ): # columns for multipole moments
#        calcAllMoments_inputArgs[i][j] = tuple( calcAllMoments_inputArgs[i][j] )
#        
#    
#import itertools as itr   
## this magic creates the matrix by "repeating" a numpy vector into a list and then "repeating" that list to give a 2D matrix of numpy vectors
#allMomentsPredicted = list(itr.repeat(list(itr.repeat( np.zeros(numTest),numAtoms*2)),numMoments-1))
#         
## Initiate multithreaded pool with cpu_count() with "starmap" routine     
##           For an example of starmap for multicore processing see:  
##           http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
#
### Initialise a pool 
#with Pool(cpu_count()) as workPool:
#    for i in range(0,numMoments-1):
#        # Formatted printing to determine calculation progress   Docs:  https://docs.python.org/3/tutorial/inputoutput.html
#        print("Predicting moment {} \t {:.0f}% done".format( (i+1), ((i+1)/numMoments)*100 ) )
#        
#        # Do work with pool instance, put the results in allMomentsPredicted, use calcAllMoments function and calcAllMoments_inputArgs as input
#        allMomentsPredicted[i] = workPool.starmap( calcAllMoments, calcAllMoments_inputArgs[i] )
#
## Details for stopping the pool's parallel execution
#workPool.close()
#workPool.join()
#print("Done")