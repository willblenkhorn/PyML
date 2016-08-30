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
"""
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns # nice theme <<< OPTIONAL >>

numTest = 2000 # number of test data points

# input data for each atom
O1rawData = np.loadtxt(open("O1_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )
H2rawData = np.loadtxt(open("H2_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )
H3rawData = np.loadtxt(open("H3_TRAINING_SET_labelled_notop2.csv","rb"),delimiter=",",skiprows=1 )

# Selected the valid moments, which change appreciably and excluded non-descriptive ones.
O1moments = np.column_stack( (O1rawData[:,3], O1rawData[:,5:8], 
                              O1rawData[:,10:12], O1rawData[:,13:15], 
O1rawData[:,17:20], O1rawData[:,22:24], O1rawData[:,26:28]) )

H2moments = np.column_stack( (H2rawData[:,3], H2rawData[:,5:8], 
                              H2rawData[:,10:12], H2rawData[:,13:15], 
H2rawData[:,17:20], H2rawData[:,22:24], H2rawData[:,26:28]) )

H3moments = np.column_stack( (H3rawData[:,3], H3rawData[:,5:8], 
                              H3rawData[:,10:12], H3rawData[:,13:15], 
H3rawData[:,17:20], H3rawData[:,22:24], H3rawData[:,26:28]) )

geometry = O1rawData[:,:3]


#-------------------------------------------------
# Model training code


# monopole model needs to, predict oxygen from the coordinates, then predict hydrogen's monopole from 
# oxygen's monopole plus the coordinates

#-------------------------------------------------


#     TODO: 
#-> fine grained paramerisation of the C, gamma and epsilon terms in a series of range based for loops looking at the error for a given parameter.
#-> function to select columns in input which have a mean and stdev <1E-5 (or programmable threshold) 
#-> Hydrogens monopole sum is the same magnitude and opposite sign as the oxygen atom (may save some computation)

# Model to predict each atom's monopole

O1_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
H2_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
H3_q00_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 

# multipole names 			(without excluded multipoles) 
# Q[0,0] monopole, Q[1,x] dipole, Q[2,x] quadrupole, Q[3,x] octopole, ...

# multipole symbols 		(without excluded multipoles)
# Q[0,0], Q[1,1,c], Q[1,1,s], Q[2,0], Q[2,2,c], Q[2,2,s], Q[3,1,c], Q[3,1,s], 
# Q[3,3,c], Q[3,3,s], Q[4,0], Q[4,2,c], Q[4,2,s], Q[4,4,c], Q[4,4,s]


# The training data's outputs to train with, so they should be a vector without multiple columns
O1_q00_TrainOut = O1moments[:-1*numTest, 0] # Oxygen model's testSetData output   
H2_q00_TrainOut = H2moments[:-1*numTest, 0] # h2 test data
H3_q00_TrainOut = H3moments[:-1*numTest, 0] # h3 test data

## Input training data 
O1_q00_TrainIn = geometry[:-1*numTest,:] # geometry only
H2_q00_TrainIn = np.column_stack(( O1_q00_TrainIn, O1moments[:-1*numTest,0] )) # geometry + oxygen's monopole
#H3_q00_TrainIn = np.column_stack(( H2_q00_TrainIn, H2moments[:-1*numTest,0] )) # geometry + O1 mono + H2 mono
H3_q00_TrainIn = H2_q00_TrainIn # geometry + O1 mono + H2 mono


# all the geometries are included (implicitly) in the bond lengths and the bond angle
# it's a simplification to putting all the xyz coordinates, "compression" of that information
# spherical polar representation with two vectors


#------------------------------------------------------------
# Construct model, using ideallised, precalculated data
#------------------------------------------------------------
O1_q00_Model.fit( O1_q00_TrainIn, O1_q00_TrainOut ) 
H2_q00_Model.fit( H2_q00_TrainIn, H2_q00_TrainOut )
H3_q00_Model.fit( H3_q00_TrainIn, H3_q00_TrainOut )


#==========================================================================================
# Model usage, to predict successive quantities
#==========================================================================================
#       Ideallised validation set data:
#       Use these ideallised values to validate the model is working well for error analysis later :)
#
#       O1_q00_TestIn = geometry[-1*numTest:,:] # geometry only
#       H2_q00_TestIn = np.column_stack(( O1moments[-1*numTest:,0], O1_q00_TestIn )) # geometry + oxygen's monopole
#       H3_q00_TestIn = np.column_stack(( H2moments[-1*numTest:,0], H2_q00_TestIn  )) # geometry + O1 mono + H2 mono
#
#       Use ideallised model to predict "ideal" values using test set
#       O1_q00_predict = O1_q00_Model.predict( O1_q00_TestIn )
#       H2_q00_predict = H2_q00_Model.predict( H2_q00_TestIn )
#       H3_q00_predict = H3_q00_Model.predict( H3_q00_TestIn )
#
#
#--------------------------------------------------------------------------------------

# oxygen monopole model uses geometry only to predict
O1_q00_TestIn = geometry[-1*numTest:,:] 
O1_q00_predicted = O1_q00_Model.predict( O1_q00_TestIn )

# Use predicted oxygen monopole in H2_q00_TestIn, input for H2's prediction
H2_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted )) # geometry + oxygen's PREDICTED monopole
H2_q00_predicted = H2_q00_Model.predict( H2_q00_TestIn )

# Use predicted O1 and H2 monopole in H3_q00_TestIn, input for H3's prediction
#H3_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted, H2_q00_predicted )) # geometry + O1 mono (predicted) + H2 mono (predicted)
H3_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted )) # geometry + O1 mono (predicted) + H2 mono (predicted)
H3_q00_predicted = H3_q00_Model.predict( H3_q00_TestIn ) # These are indeed a 1D matrix as long as the test set


#===================================================================================
# graph to show conservation of monopole moment is preserved in the prediction model
#-----------------------------------------------------------------------------------
sumHX_q00_predicted = H3_q00_predicted + H2_q00_predicted # These are the predicted values

sumHXmoments = H2moments + H3moments        # These are the actual calculated values

plt.scatter( sumHXmoments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
plt.scatter(sumHX_q00_predicted, O1_q00_predicted, c='b', s=8, label='predicted')
plt.xlabel( 'Sum Hydrogen 2 & 3 q00' )
plt.ylabel( 'Oxygen 1 q00' )
plt.title( 'O1 q00 vs. Sum Hydrogen\'s q00' )
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


#===================================================================================
# This is the reference model which only uses input geometry,
#
#       oxygen will not experience a benefit, since that was calculated first in the new
#       model, so isn't compared here.
#-----------------------------------------------------------------------------------
H2_q00_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
H3_q00_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 

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


#=================================================================================
# A H2_q00 log-linear histogram of the monopole's error, for the new and old model
#---------------------------------------------------------------------------------
# calculate distribution of error in the new and older model
H2_q00_ErrorDist = np.abs( np.abs( H2_q00_predicted ) - np.abs( H2moments[-1*numTest:,0] ) ) 
H2_q00_ErrorDistOld = np.abs( np.abs( H2_q00_predictedOld ) - np.abs( H2moments[-1*numTest:,0] ) ) 

fig = plt.figure()
MIN, MAX = .00001, 0.01 # Define the range on the graph's axis

n, bins, patches = plt.hist(H2_q00_ErrorDist, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label='Moments and geometry' )

n, bins, patches = plt.hist(H2_q00_ErrorDistOld, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1,histtype='step', cumulative=True,color='b', linewidth=2, label='Geometry only' )

plt.gca().set_xscale("log")

plt.xlabel('Magnitude of H2_q00 Error')
plt.ylabel('Cumulative Number Fraction')
plt.title('Cumulative H2_q00 Error Distribution')
plt.grid(True)
plt.ylim(0, 1.05)

legend = plt.legend(loc='upper left', shadow=True, fontsize='large')

fig.savefig('cumulative_H2_q00_error.png',dpi=600)
plt.show()

H2_q00_avError = np.mean(H2_q00_ErrorDist)
H2_q00_avErrorOld = np.mean(H2_q00_ErrorDistOld)

percH2Improved = (H2_q00_avErrorOld - H2_q00_avError) / H2_q00_avError * 100

print("The average improvement from including other moments is ", percH2Improved, "%" )

#=================================================================================
# A H3_q00 log-linear histogram of the monopole's error, for the new and old model
#---------------------------------------------------------------------------------
# calculate distribution of error in the new and older model
H3_q00_ErrorDist = np.abs( np.abs( H3_q00_predicted ) - np.abs( H3moments[-1*numTest:,0] ) ) 
H3_q00_ErrorDistOld = np.abs( np.abs( H3_q00_predictedOld ) - np.abs( H3moments[-1*numTest:,0] ) ) 

fig = plt.figure()
MIN, MAX = .00001, 0.01 # Define the range on the graph's axis

n, bins, patches = plt.hist(H3_q00_ErrorDist, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label='Moments and geometry' )

n, bins, patches = plt.hist(H3_q00_ErrorDistOld, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1,histtype='step', cumulative=True,color='b', linewidth=2, label='Geometry only' )

plt.gca().set_xscale("log")
plt.xlabel('Magnitude of H3_q00 Error')
plt.ylabel('Cumulative Number Fraction')
plt.title('Cumulative H3_q00 Error Distribution')
plt.grid(True)
plt.ylim(0, 1.05)

legend = plt.legend(loc='upper left', shadow=True, fontsize='large')

fig.savefig('cumulative_H3_q00_error.png',dpi=600)
plt.show()

H3_q00_avError = np.mean(H3_q00_ErrorDist)
H3_q00_avErrorOld = np.mean(H3_q00_ErrorDistOld)

percH3Improved = (H3_q00_avErrorOld - H3_q00_avError) / H3_q00_avError * 100

print("The average improvement from including other moments is ", percH3Improved, "%" )
######################################################################

#===================================================================================
# graph to show conservation of monopole moment is NOT preserved in the OLD prediction model
#-----------------------------------------------------------------------------------
sumHX_q00_predictedOld = H2_q00_predictedOld + H2_q00_predictedOld # These are the predicted values
sumHXmoments = H2moments + H3moments        # These are the actual calculated values

plt.scatter( sumHXmoments[-1*numTest:,0], O1moments[-1*numTest:,0], c='r', s=8, label='calculated')
# NB the prediction of oxygen is from geometry in both cases
plt.xlabel( 'Sum Hydrogen 2 & 3 q00 from old model' )
plt.scatter(sumHX_q00_predictedOld, O1_q00_predicted, c='b', s=8, label='predicted in old model') 
plt.ylabel( 'Oxygen 1 q00' )
plt.title( 'O1 q00 vs. Sum Hydrogen\'s q00 in OLD model' )
plt.legend()
plt.show()

#==============================================================================
# Dipole prediction on the oxygen atom
#   Predicted before hydrogen dipoles
#   relies on geometry and oxygen charge
#   Optionally try to include hydrogen charge and see if it improves the result
#------------------------------------------------------------------------------

O1_q11c_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
H2_q11c_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 
H3_q11c_Model = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 
O1_q11c_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
H2_q11c_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 
H3_q11c_ModelOld = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 ) 

# The training data outputs to train with, so they should be a vector without multiple columns
#   These are unchanged but redefined for clarity
O1_q11c_TrainOut = O1moments[:-1*numTest, 1]
H2_q11c_TrainOut = H2moments[:-1*numTest, 1] # h2 test data
H3_q11c_TrainOut = H3moments[:-1*numTest, 1] # h3 test data

## Input training data 
O1_q11c_TrainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
H2_q11c_TrainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
H3_q11c_TrainIn = np.column_stack(( geometry[:-1*numTest,:], O1moments[:-1*numTest, 0] ))
O1_q11c_TrainInOld = geometry[:-1*numTest,:]
H2_q11c_TrainInOld = geometry[:-1*numTest,:]
H3_q11c_TrainInOld = geometry[:-1*numTest,:]

#------------------------------------------------------------
# Construct model, using ideallised, precalculated data
#------------------------------------------------------------
O1_q11c_Model.fit( O1_q11c_TrainIn, O1_q11c_TrainOut )
H2_q11c_Model.fit( H2_q11c_TrainIn, H2_q11c_TrainOut )
H3_q11c_Model.fit( H3_q11c_TrainIn, H3_q11c_TrainOut )
O1_q11c_ModelOld.fit( O1_q11c_TrainInOld, O1_q11c_TrainOut )
H2_q11c_ModelOld.fit( H2_q11c_TrainInOld, H2_q11c_TrainOut )
H3_q11c_ModelOld.fit( H3_q11c_TrainInOld, H3_q11c_TrainOut )
#------------------------------------------------------------
# Using the model
#------------------------------------------------------------
O1_q11c_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted ))
H2_q11c_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted ))
H3_q11c_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted ))
O1_q11c_TestInOld = geometry[-1*numTest:,:] 
H2_q11c_TestInOld = geometry[-1*numTest:,:] 
H3_q11c_TestInOld = geometry[-1*numTest:,:] 

O1_q11c_predicted = O1_q11c_Model.predict( O1_q11c_TestIn )
H2_q11c_predicted = H2_q11c_Model.predict( H2_q11c_TestIn )
H3_q11c_predicted = H3_q11c_Model.predict( H3_q11c_TestIn )
O1_q11c_predictedOld = O1_q11c_ModelOld.predict( O1_q11c_TestInOld )
H2_q11c_predictedOld = H2_q11c_ModelOld.predict( H2_q11c_TestInOld )
H3_q11c_predictedOld = H3_q11c_ModelOld.predict( H3_q11c_TestInOld )

#=================================================================================
# A O1_q11c log-linear histogram of the monopole's error, for the new and old model
#---------------------------------------------------------------------------------
# calculate distribution of error in the new and older model
O1_q11c_ErrorDist = np.abs( np.abs( O1_q11c_predicted ) - np.abs( O1moments[-1*numTest:,1] ) ) 
O1_q11c_ErrorDistOld = np.abs( np.abs( O1_q11c_predictedOld ) - np.abs( O1moments[-1*numTest:,1] ) ) 

fig = plt.figure()
MIN, MAX = np.min(O1_q11c_ErrorDist), np.max(O1_q11c_ErrorDist) # Define the range on the graph's axis

n, bins, patches = plt.hist(O1_q11c_ErrorDist, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label='Moments and geometry' )

n, bins, patches = plt.hist(O1_q11c_ErrorDistOld, 
                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
normed=1,histtype='step', cumulative=True,color='b', linewidth=2, label='Geometry only' )

plt.gca().set_xscale("log")
plt.xlabel('Magnitude of O1_q11c Error')
plt.ylabel('Cumulative Number Fraction')
plt.title('Cumulative O1_q11c Error Distribution')
plt.grid(True)
plt.ylim(0, 1.05)

legend = plt.legend(loc='upper left', shadow=True, fontsize='large')

fig.savefig('cumulative_O1_q11c_error.png',dpi=600)
plt.show()

O1_q11c_avError = np.mean(O1_q11c_ErrorDist)
O1_q11c_avErrorOld = np.mean(O1_q11c_ErrorDistOld)

percO1_q11c_Improved = (O1_q11c_avErrorOld - O1_q11c_avError) / O1_q11c_avError * 100

print("The average improvement from including other moments is ", percO1_q11c_Improved, "%" )

#===========================================================

#def printAverageError( predicted, actual ):
#	
#	error = np.mean( np.abs( np.abs(predicted) - np.abs(actual)))
#	print( "the mean squared error on the monopole is", error)
#	MeanRaw = np.abs( np.mean( O1rawData[-1*numTest:,3] ) )
#	print( "the average percentage error is therefore", (error / MeanRaw ) * 100 )
#
#	return 0 
#	
#def calculateAveragePercentageError( predicted, actual ):
#	
#	error = np.mean( np.abs( np.abs(predicted) - np.abs(actual))) ## Should we instead use the distribution instead of just the mean?
#	MeanRaw = np.abs( np.mean( O1rawData[-1*numTest:,3] ) )
#	
#	return	( error / MeanRaw ) * 100
