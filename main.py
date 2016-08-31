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
    MIN, MAX = .00001, 0.1
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

def debugPrintFns():
    #===================================================================================
    # graph to show conservation of monopole moment is preserved in a good model
    #-----------------------------------------------------------------------------------
    sumHX_q00_predictedOld = H2_q00_predictedOld + H2_q00_predictedOld # These are the predicted sums (geom)
    sumHX_q00_predicted = H3_q00_predicted + H2_q00_predicted # These are the predicted sums (geom + moments)
    sumHXmoments = H2moments + H3moments                      # These are the "true" calculated values
    plt.scatter(sumHX_q00_predictedOld, O1_q00_predicted, c='g', s=8, label='Predicted from geometry') 
    plt.scatter( sumHXmoments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='Calculated')
    plt.scatter( sumHX_q00_predicted, O1_q00_predicted, c='b', s=8, label='Predicted from monopole and geometry')
    
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
    plt.scatter(H2_q00_predicted, O1_q00_predicted, c='b', s=8, label='predicted')
    plt.xlabel( 'Hydrogen 2 q00' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. Hydrogen 2 q00' )
    plt.legend()
    plt.show()
    #===================================================================================
    # graph to show O1 vs H3 monopole moment is predicted well by model
    #-----------------------------------------------------------------------------------
    plt.scatter( H3moments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
    plt.scatter(H3_q00_predicted, O1_q00_predicted, c='b', s=8, label='predicted')
    plt.xlabel( 'Hydrogen 3 q00' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. Hydrogen 3 q00' )
    plt.legend()
    plt.show()
    #===================================================================================
    # graph to show O1 monopole moment vs bondlen1 is predicted well by model
    #-----------------------------------------------------------------------------------
    plt.scatter( geometry[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
    plt.scatter( geometry[-1*numTest:,0], O1_q00_predicted, c='b', s=8, label='predicted')
    plt.xlabel( 'bond length1' )
    plt.ylabel( 'Oxygen 1 q00' )
    plt.title( 'O1 q00 vs. bond length1' )
    plt.legend()
    plt.show()
    #===================================================================================

if __name__ == '__main__':
    freeze_support() # Windows specific, multithreaded crash handling
    numTest = 2000 # number of test data points out of the total input data of 3997 points  
    #===============================================================================
    # TODO: Create input data processesing function to select columns in input 
    #       which have a mean and stdev <1E-5 (or programmable threshold)
    
    
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
    #===============================================================================
    #     TODO: 
    #-> fine grained paramerisation of the C, gamma and epsilon terms in a series of range based for loops looking at the error for a given parameter.
    #===============================================================================
    
    #===============================================================================
    # Model training code
    #       monopole model needs to, predict oxygen from the coordinates, then predict 
    #       hydrogen's monopole from oxygen's monopole plus the coordinates
    #===============================================================================
    
    # all the geometries are included (implicitly) in the bond lengths and the bond angle
    # it's a simplification to putting all the xyz coordinates, "compression" of that information
    # spherical polar representation with two vectors

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
    O1_q00_model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 ) 
    O1_q00_trainIn = geometry[:-1*numTest,:]
    O1_q00_trainOut = O1moments[:-1*numTest, 0]
    O1_q00_testIn = geometry[-1*numTest:,:] 
    O1_q00_model.fit(O1_q00_trainIn, O1_q00_trainOut)
    O1_q00_predicted = O1_q00_model.predict(O1_q00_testIn)
            
#   Input arguments for multi threading "starmap" routine           
    q00_inputArgs = [("H2","new"),("H2","old"),("H3","new"),("H3","old")]
    q00_result = [np.zeros(numTest), np.zeros(numTest), np.zeros(numTest),
              np.zeros(numTest)]
              
              
#   Multithreaded "starmap" section, each fn call passes two args
    with Pool(cpu_count()) as workPool:
        q11c_result = workPool.starmap(monopoleModel, q00_inputArgs)
        workPool.close()
        workPool.join()
    # Result now contains:  O1_old, H2_new, H2_old, H3_new, H3_old
#    q00_result = np.column_stack((O1_q00_predicted, q00_result))

## figure out how to put things to the start of the list




    
    O1_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 )
    H2_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 )
    H3_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 ) 
    
    # TODO: Hydrogen's monopole sum is the same magnitude and opposite sign as the oxygen atom,
    #       use this to avoid the computation of H3 by: O1 - H2 = H3  This is quite a specific case though...
    #       Q: How could that be generalised?
    #       A: Could look at the prediction accuracy, if low can try to deduce? Difficult to generalise and if >1
    
    # The training data's outputs to train with, so they should be a vector without multiple columns
    O1_q00_TrainOut = O1moments[:-1*numTest, 0] # Oxygen model's testSetData output   
    H2_q00_TrainOut = H2moments[:-1*numTest, 0] # h2 test data
    H3_q00_TrainOut = H3moments[:-1*numTest, 0] # h3 test data
    
    ## Input training data 
    O1_q00_TrainIn = geometry[:-1*numTest,:] # geometry only
    H2_q00_TrainIn = np.column_stack(( O1_q00_TrainIn, O1moments[:-1*numTest,0] )) # geometry + oxygen's monopole
    H3_q00_TrainIn = H2_q00_TrainIn # geometry + Oxygen's monopole only
    
    # Surprisingly if you run it with the additional information from the H2 monopole, it performs worse not better!
    #   H3_q00_TrainIn = np.column_stack(( H2_q00_TrainIn, H2moments[:-1*numTest,0] )) # geometry + O1 mono + H2 mono
    
    #------------------------------------------------------------
    # Construct model, using ideallised, precalculated data
    #------------------------------------------------------------
    O1_q00_Model.fit( O1_q00_TrainIn, O1_q00_TrainOut ) 
    H2_q00_Model.fit( H2_q00_TrainIn, H2_q00_TrainOut )
    H3_q00_Model.fit( H3_q00_TrainIn, H3_q00_TrainOut )
    
    #==========================================================================================
    # Model usage, to predict successive quantities
    #==========================================================================================
    
    # oxygen monopole model uses geometry only, in prediction
    O1_q00_TestIn = geometry[-1*numTest:,:] 
    O1_q00_predicted = O1_q00_Model.predict( O1_q00_TestIn )
    
    # Use predicted oxygen monopole in H2_q00_TestIn, input for H2's prediction
    H2_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted )) # geometry + oxygen's PREDICTED monopole
    H2_q00_predicted = H2_q00_Model.predict( H2_q00_TestIn )
    
    # Using predicted O1 and H2 monopole in H3_q00_TestIn, input for H3's prediction reduced accuracy
    #H3_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted, H2_q00_predicted )) # geometry + O1 mono (predicted) + H2 mono (predicted)
    
    # Using just geometry and O1 monopole increased accuracy, though not by much (3%)
    H3_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted )) # geometry + O1 mono (predicted) 
    H3_q00_predicted = H3_q00_Model.predict( H3_q00_TestIn ) # These are indeed a 1D matrix as long as the test set
    
    
    #===================================================================================
    # This is the reference model which only uses input geometry,
    #
    #       oxygen will not experience a benefit, since that was calculated first in the new
    #       model, so isn't compared here.
    #-----------------------------------------------------------------------------------
    H2_q00_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 )
    H3_q00_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 ) 
    
    # The training data outputs to train with, so they should be a vector without multiple columns
    #   These are unchanged but redefined for clarity
    H2_q00_TrainOutOld = H2moments[:-1*numTest, 0] # h2 test data
    H3_q00_TrainOutOld = H3moments[:-1*numTest, 0] # h3 test data
    
    ## Input training data 
    H2_q00_TrainInOld = geometry[:-1*numTest,:] # geometry only
    H3_q00_TrainInOld = geometry[:-1*numTest,:] # geometry only
    
    #------------------------------------------------------------
    # Construct model, using ideallised, precalculated data
    #------------------------------------------------------------
    H2_q00_ModelOld.fit( H2_q00_TrainInOld, H2_q00_TrainOutOld )
    H3_q00_ModelOld.fit( H3_q00_TrainInOld, H3_q00_TrainOutOld )
    #------------------------------------------------------------
    # Using the reference model
    #------------------------------------------------------------
    H2_q00_TestInOld = geometry[-1*numTest:,:] 
    H3_q00_TestInOld = geometry[-1*numTest:,:] 
    H2_q00_predictedOld = H2_q00_ModelOld.predict( H2_q00_TestInOld )
    H3_q00_predictedOld = H3_q00_ModelOld.predict( H3_q00_TestInOld )
    
    #-----------------------------------------------------------------------------------------
    # A H2_q00 & H3_q00 log-linear histogram of the monopole's error, for the new and old model
    #    printError( H2_q00_predicted, H2_q00_predictedOld, H2moments[-1*numTest:,0], 'H2_q00' )
    #    printError( H3_q00_predicted, H3_q00_predictedOld, H3moments[-1*numTest:,0], 'H3_q00' )
    #-----------------------------------------------------------------------------------------

    def dipoleModel(atom, newOrOld):
#   Purpose: Calculates the dipole value
#   Usage: Accepts atom type and if new or old model and calculates the dipole value.
        model = SVR( kernel='rbf', C=5E3, gamma=0.001, epsilon =0.001 )    
        if( newOrOld == "new" ):
            if(atom == "O1"):
                trainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
                trainOut = O1moments[:-1*numTest, 1]
                testIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted ))
            if(atom == "H2"):
                trainIn = np.column_stack(( geometry[:-1*numTest,:], H2moments[:-1*numTest, 0] ))
                trainOut = H2moments[:-1*numTest, 1]
                testIn = np.column_stack(( geometry[-1*numTest:,:], H2_q00_predicted )) 
            if(atom == "H3"):
                trainIn = np.column_stack(( geometry[:-1*numTest,:], H3moments[:-1*numTest, 0] ))
                trainOut = H3moments[:-1*numTest, 1]
                testIn = np.column_stack(( geometry[-1*numTest:,:], H3_q00_predicted ))
            model.fit(trainIn, trainOut)
            return model.predict(testIn)
                
        if(newOrOld=="old"):
            trainIn = geometry[:-1*numTest,:]
            testIn = geometry[-1*numTest:,:]
            if(atom == "O1"):
                trainOut = O1moments[:-1*numTest, 1]
            if(atom == "H2"):
                trainOut = H2moments[:-1*numTest, 1]
            if(atom == "H3"):
                trainOut = H3moments[:-1*numTest, 1]  
            model.fit(trainIn, trainOut)
            return model.predict(testIn)
            
#   Input arguments for multi threading "starmap" routine     
#           For an example of starmap for multicore processing see:  
#           http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
    q11c_inputArgs = [("O1","new"),("O1","old"),("H2","new"),("H2","old"),("H3","new"),("H3","old")]
    q11c_result = [np.empty(numTest), np.empty(numTest), np.empty(numTest),
              np.empty(numTest), np.empty(numTest), np.empty(numTest)]
              
#   Multithreaded "starmap" section, each fn call passes two args
    with Pool(cpu_count()) as workPool:
        q11c_result = workPool.starmap(dipoleModel, q11c_inputArgs)
        workPool.close()
        workPool.join()

#    printError( q11c_result[0], q11c_result[1], O1moments[-1*numTest:,1], 'O1_q11c' )
#    printError( q11c_result[2], q11c_result[3], H2moments[-1*numTest:,1], 'H2_q11c' )
#    printError( q11c_result[4], q11c_result[5], H3moments[-1*numTest:,1], 'H3_q11c' )

    