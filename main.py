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
O1moments = np.column_stack( (O1rawData[:,3], O1rawData[:,5:8], O1rawData[:,10:12], O1rawData[:,13:15], O1rawData[:,17:20], O1rawData[:,22:24], O1rawData[:,26:28]) )

H2moments = np.column_stack( (H2rawData[:,3], H2rawData[:,5:8], H2rawData[:,10:12], H2rawData[:,13:15], H2rawData[:,17:20], H2rawData[:,22:24], H2rawData[:,26:28]) )

H3moments = np.column_stack( (H3rawData[:,3], H3rawData[:,5:8], H3rawData[:,10:12], H3rawData[:,13:15], H3rawData[:,17:20], H3rawData[:,22:24], H3rawData[:,26:28]) )

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
# Q[0,0] monopole, Q[1,x] dipole, Q[2,x] quadrupole, Q[3,x]

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
H3_q00_TrainIn = np.column_stack(( H2_q00_TrainIn, H2moments[:-1*numTest,0] )) # geometry + O1 mono + H2 mono


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
H3_q00_TestIn = np.column_stack(( geometry[-1*numTest:,:], O1_q00_predicted, H2_q00_predicted )) # geometry + O1 mono (predicted) + H2 mono (predicted)
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

"""
BELOW NEEDS WORK, I.E. TO BE COMPLETED
"""

#def print_average_error(predicted,actual):
#	
#	error = np.mean( np.abs( np.abs(predicted) - np.abs(actual)))
#	print( "the mean squared error on the monopole is", error)
#	MeanRaw = np.abs( np.mean( O1rawData[-1*numTest:,3] ) )
#	print( "the average percentage error is therefore", (error / MeanRaw ) * 100 )
#
#	return 0 
#	
#def calculate_average_percentage_error(predicted,actual):
#	
#	error = np.mean( np.abs( np.abs(predicted) - np.abs(actual))) ## Should we instead use the distribution instead of just the mean?
#	MeanRaw = np.abs( np.mean( O1rawData[-1*numTest:,3] ) )
#	
#	return	( error / MeanRaw ) * 100
#	
#
## so we need to create the models and then move onto testing the "prediction" tree
## input -> monopole(o)
## input + monopole(o) -> monopole(H)....
#
#
#
#
#
## Calculate error for the monopole
# = np.mean(  np.abs( np.abs(oxgyenMonopolePredicted) - np.abs(O1rawData[-1*numTest:,3]) )  )
#print( "the mean squared error on the monopole is", OxygenMonopoleError)
#oxygenMeanRaw = np.abs( np.mean( O1rawData[-1*numTest:,3] ) )
#print( "the percentage error is therefore", (OxygenMonopoleError / oxygenMeanRaw ) * 100 )
#
#
#
#
#
#
#
#
#
#
## Fit model to predict dipole from input geometry and monopole
#dipoleModel = SVR( kernel='rbf', C=5E3, gamma=0.001, cache_size=1600, epsilon =0.001 )
##dipoleModel = SVR( kernel='poly', C=1, degree=6, gamma=0.35, cache_size=1600, epsilon =0.012 ) # Polynomial fit
#
## positions and monopole used to train the dipole moment
#posMonoTraining = O1rawData[:-1*numTest,:4]  
## Considering dipole (moment3), since it has a trend with position, easier to test.
#dipoleTraining = O1rawData[:-1*numTest, 5] 
## Constructing model  
#dipoleModel.fit( posMonoTraining, dipoleTraining )
## Constructing test set
#dipoleTestSet = O1rawData[-1*numTest:,:4]
#
#
## Predict using model
#dipolePredicted = dipoleModel.predict( dipoleTestSet )
#
#
#
#
#
#
#
## Calculate error for the dipole
#dipoleError = np.mean(  np.abs( np.abs(dipolePredicted) - np.abs(O1rawData[-1*numTest:,5]) )  ) 
#print( "the mean squared error on the dipole is", dipoleError)
#meanDipoleRaw = np.abs( np.mean( dipoleTestSet ) )
#print( "the percentage error is therefore", ( dipoleError / meanDipoleRaw ) * 100 )
#
#
#######################################################################
## the histogram of the monopole's and dipole's error
#######################################################################
#
## calculate distribution of error instead of just the average
#monpoleErrorDist = np.abs( np.abs(monopolePredicted) - np.abs(O1rawData[-1*numTest:,3]) ) 
#
#fig = plt.figure()
#MIN, MAX = .0001, 0.015 # Define the range on the graph's axis
#
#n, bins, patches = plt.hist(monpoleErrorDist, 
#                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
#normed=1, histtype='step', cumulative=True, color='r', linewidth=2, label='monopole' )
#                            
##pl.hist(data, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))
#plt.gca().set_xscale("log")
#
## add a 'best fit' line
##y = mlab.normpdf( bins, mu, sigma)
##l = plt.plot(bins, y, 'r--', linewidth=1)
#
#plt.xlabel('Magnitude of Error')
#plt.ylabel('Probability')
#plt.title('Cumulative Error Distribution')
##plt.axis([40, 160, 0, 0.03])
#plt.grid(True)
#plt.ylim(0, 1.05)
##plt.show()
#################################
#
#
#dipoleErrorDist = np.abs( np.abs(dipolePredicted) - np.abs(O1rawData[-1*numTest:,5]) ) 
#
##plt.figure()
#MIN, MAX = .0001, 0.015
#
#n, bins, patches = plt.hist(dipoleErrorDist, 
#                            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50),
#normed=1,histtype='step', cumulative=True,color='b', linewidth=2, label='Dipole' )
#
#legend = plt.legend(loc='center right', shadow=True, fontsize='large')
#                            
#
## add a 'best fit' line
##y = mlab.normpdf( bins, mu, sigma)
##l = plt.plot(bins, y, 'r--', linewidth=1)
#
#
#fig.savefig('cumulative_error.png',dpi=600)
#plt.show()
#######################################################################
