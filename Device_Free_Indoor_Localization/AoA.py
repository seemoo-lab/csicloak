#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides an algorithm for AoA calculation based on the matlab implementation of:

DOA estimation based on MUSIC algorithm
https://pdfs.semanticscholar.org/5ff7/806b44e60d41c21429e1ad2755d72bba41d7.pdf

"""

# --------------------------------------- IMPORTS ---------------------------------------- #

import enumerators
import visualize

import matplotlib.pyplot as plt
import numpy as np
import math

# ----------------------------------- HELPER METHOD -------------------------------------- #

def getMaxDiff(array):
	return abs(np.max(array) - np.min(array))

# --------------------- MUSIC ALGO BASED ON GIVEN SOURCE ABOVE --------------------------- #

def applyMUSIC(data, distance, wavelength, plotData, printRes):

	# Store the result
	result = np.zeros([data.shape[0]], dtype=complex);

	# Calculate the angle for each given spatialstream
	for spatialstream in range(0,data.shape[0]):
  		
  		# Only use 3 of 4 antennas (not the internal one)
		x = np.zeros([3,data.shape[2]], dtype=complex)
  
  		# Order of the antennas: Right (0) - Middle (3) - Left (1)
		x[0,:] = data[spatialstream,0,:]
		x[1,:] = data[spatialstream,3,:]
		x[2,:] = data[spatialstream,1,:]	

		# M = Amount of antennas in RX-array
		# P = Amount of signals to analyze -> Just one direction  
		M = 3
		P = 1

		# Calculate covariance Matrix
		R = x.dot(np.transpose(np.conj(x)))

		# Find Eigenvalues and Eigenvectors of R
		V_arr,N = np.linalg.eig(R)

		# Sort it
		sort_perm = (V_arr).argsort()
		V_arr = V_arr[sort_perm]
		
		# Biggest ones = Signal; Rest = Noise subspace
		N = N[:, sort_perm]

		# V -> Array to Diagonal Matrix
		V = np.zeros([3,3], dtype=complex)
		np.fill_diagonal(V,V_arr)

		# Estimate noise subspace
		NN=N[:,0:M-P]

		# Search space
		theta = np.arange(-90,90,0.5)
		Pmusic = np.zeros([len(theta)], dtype=complex)

		# Calculate the angle
		for ii in range(0,len(theta)):
			SS=np.zeros([M], dtype=complex)
			for jj in range(0,M):
 				SS[jj]=np.exp(-1j*2*jj*np.pi*distance*np.sin(theta[ii]/180*np.pi)/wavelength)
			
			PP=SS.dot(NN.dot((np.transpose(np.conj(NN))).dot(np.transpose(np.conj(SS)))))

			Pmusic[ii]=abs(1/ PP)
		
		Pmusic=10*np.log10(Pmusic/(max(Pmusic))); 

		# Get index of estimated angle
		max_val_idx = np.argmax(Pmusic)

		# Visualize all
		if(plotData == True):
			plt.plot(theta, Pmusic,'-k')
			plt.title("DOA estimation based on MUSIC algorithm")
			plt.xlabel("angle theta/degree")
			plt.ylabel("spectrum function P(theta) /dB")
			plt.grid(b=True, which='major', color='#666666', linestyle='-')
			plt.show()

		if(printRes):
			print("Stream %d: The estimated angle is: %.2f° " % (spatialstream,theta[max_val_idx]))
		
		# Store the result
		result[spatialstream] = theta[max_val_idx]

	return result

# ---------------------------------------------------------------------------------------- #