#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides general functions for data handling, printing, small helper functions as well as logging functionality

"""

# --------------------------------------------- IMPORTS ------------------------------------------------------ #

from datetime import datetime

import pickle

import numpy as np

import time

import enumerators

from parameters import storeDirectory, modelDir, loadModelAsNet, loadRefinedNetwork

from colored import fg, bg, attr		# More information: https://pypi.org/project/colored/


########################################## Helper methods ######################################################

# ---------------------------------------- File Handling ----------------------------------------------------- #

# Load file
def loadFile(name):

	# Models are stored in an extra subfolder!
	if ("model" in name and not loadRefinedNetwork) or ("R_" in name and len(loadModelAsNet) >= 1):
		directory = storeDirectory + "models/" + modelDir
		print("Search in model specfic folder: %s" % (directory))
	else:
		directory = storeDirectory

	path = directory + name

	# Load the requested file
	filehandler = open(path,'rb')
	obj = pickle.load(filehandler)
	filehandler.close()
	return obj

# Save file
def saveFile(name, obj):

	# Store the given obj on the place given in name
	path = storeDirectory + name
	filehandler = open(path,'wb')
	pickle.dump(obj, filehandler)
	filehandler.close()

# Create filename following specific pattern
def createFileName(dataID, netID, lossID, extremes, numExtremes):

	filename = ""

	# -1 means, that the property is not relevant; e.g. data is only stored with dataID, no netID necessary
	if netID == -1 and lossID == -1 and extremes == -1:
		filename = "data_%s.obj" % (dataID)
	elif(extremes == -1):
		filename = "net_n%s_l%s.obj" % (netID, lossID)
	else:
		if(extremes == 0):	#Worst or best ones for extreme estimates
			filename = "worst%d_d%s_n%s_l%s.obj" % (numExtremes, dataID, netID, lossID)
		else:
			filename = "best%d_d%s_n%s_l%s.obj" % (numExtremes, dataID, netID, lossID)
	return filename

# ------------------------------------------- Print with time ------------------------------------------------ #

# Print the given text together with a timestamp
def printWithTime(text, t0):
	elapsedTime = convertTimeFormat(time.perf_counter() - t0)[:-4]
	printInfo("%s (%s)" % (text, elapsedTime))

# ------------------------------------------- Print in color ------------------------------------------------- #

# More information: https://pypi.org/project/colored/

# Print Fail with red background (black on red)
def printFailure(text):
	print ('%s%s%s%s' % (fg(0), bg(9),text,attr(0)))

# Print Success with green background (black on green)
def printSuccess(text):
	print ('%s%s%s%s' % (fg(0), bg(41), text, attr(0)))

# Print Info : just prints yellow without background
def printInfo(text):
	print ('%s%s%s' % (fg(190), text, attr(0)))

# ------------------------------------- Change Coordinate Reference ------------------------------------------ #

# So far: origin top left & indices starting with 0
# Output: origin down left (as usual) and indices starting with 1
# Also switch positions of X and Y Coordinate!
def changeCoordRef(coord, max):
    return [coord[1] + 1, max - coord[0]]

# ------------------------------------ Helper methods: Time and SC to Index ---------------------------------- #

# Convert time from seconds to HH:MM:SS.mmm
def convertTimeFormat(seconds):
	time = "%02d:%02d:%02d.%03d" % (seconds/(3600), (seconds/(60))%60, seconds%60, (seconds*1000) % 1000)
	return time

# converter methods index to subcarrier
# Map the range [0,255] into the range [-128,127]
# SC 128 = SC 0!
def indicesToSC(index):
	if(index < 128):
		sc = -1*(128-index)
	else:
		sc = index - 128
	return sc 

def scToIndices(sc):
	if(sc >= 0):
		index = sc + 128
	else:
		index = 128 - (-1*sc)
	return index

########################################## Logging methods ############################################################

def writeLog(mode, savedname, logfilename, directory, dataID, netID, lossID, time, elapsedTime, roomName, excluded, 
			officeRows, officeCols, coordInfo, coordMode, signalrepr, samplesPerTarget, 
			learningrate, momentum, batchsize, updatefreq, droprate, kernelsize, inchannels, epochsTotal, 
			lossfunctionTraining, lossfunctionTesting, numTraining, numTotal, insertedAoa, meanError, extremes, bestExtremes, 
			foundcolareas, foundrowareas, backprop, width, baseID, base_files, base_directory, base_config,
			coaxInfo, coax_directory, coaxFilesPerAntenna, dropoutEnabled, searchInTraining, selectFromSamplesPerPos,
			normalize, shuffling, rmPilots, rmUnused, usePrevRandomVal, baseNo, filterDataAfterLoading, maskID,
			randomVal, useCombinedData,amountFeatures,analyzeCDFAndLog):

	# Layout parameter 
	newline = "\n"
	separator1 = "+" + "="*(width-2) + "+"
	separator2 = "+" + "-"*(width-2) + "+"

	# Call the correct log function depending on the mode
	if(mode == enumerators.MODE.DATALOADER):
		createLogOfParametersData(dataname=savedname, logfilename=logfilename, directory=directory, dataID=dataID, time=time, 
									elapsedTime=elapsedTime, roomName=roomName, excluded=excluded, officeRows=officeRows, officeCols=officeCols, 
									coordInfo=coordInfo, coordMode=coordMode, signalrepr=signalrepr, 
									samples=numTotal, samplesPerTarget=samplesPerTarget, width=width, newline=newline, separator1=separator1, separator2=separator2,
									rmPilots=rmPilots, rmUnused=rmUnused, mask=maskID)
		print("Log written")

	elif(mode == enumerators.MODE.BASELOADER):
		createLogOfParametersData(dataname=savedname, logfilename=logfilename, directory=base_directory, dataID=baseID, time=time, 
									elapsedTime=elapsedTime, roomName=roomName, excluded=False, officeRows=enumerators.ROOM_ROW.COMPLETE, officeCols=enumerators.ROOM_COL.COMPLETE, 
									coordInfo=base_config, coordMode=coordMode, signalrepr=signalrepr, 
									samples=numTotal, samplesPerTarget=base_files, width=width, newline=newline, separator1=separator1, separator2=separator2,
									rmPilots=rmPilots, rmUnused=rmUnused, mask=maskID)
		print("Log written")

	elif(mode == enumerators.MODE.TRAINING):
		createLogOfParametersTraining(netname=savedname, logfilename=logfilename, netID=netID, dataID=dataID, lossID=lossID, time=time, 
									elapsedTime=elapsedTime, learningrate=learningrate, momentum=momentum, batchsize=batchsize,
									updatefreq=updatefreq, droprate=droprate, kernelsize=kernelsize, inchannels=inchannels, epochsTotal=epochsTotal, 
									lossfunctionTraining=lossfunctionTraining, lossfunctionTesting=lossfunctionTesting, numTraining=numTraining, 
									numTotal=numTotal, insertedAoA=insertedAoa, meanError=meanError, backprop=backprop, width=width, 
									newline=newline, separator1=separator1, separator2=separator2, dropoutEnabled=dropoutEnabled, shuffling=shuffling,
									selectFromSamplesPerPos=selectFromSamplesPerPos, normalize=normalize, usePrevRandomVal=usePrevRandomVal,
									filterDataAfterLoading=filterDataAfterLoading, randomVal=randomVal, mask=maskID, excluded=excluded, officeRows=officeRows, 
									officeCols=officeCols,useCombinedData=useCombinedData, baseNo=baseNo, amountFeatures=amountFeatures)
		print("Log written")

	elif(mode == enumerators.MODE.TESTING):
		createLogOfParametersTesting(logfilename=logfilename, netID=netID, dataID=dataID, lossID=lossID, time=time, elapsedTime=elapsedTime, lossfunction=lossfunctionTesting, 
									numTraining=numTraining, numTotal=numTotal, insertedAoA=insertedAoa, meanError=meanError, width=width, newline=newline, separator1=separator1, separator2=separator2,
									randomVal=randomVal, usePrevRandomVal=usePrevRandomVal, shuffling=shuffling, mask=maskID, baseNo=baseNo, useCombinedData=useCombinedData,
									filterDataAfterLoading=filterDataAfterLoading, excluded=excluded, officeRows=officeRows, officeCols=officeCols, normalize=normalize,
									amountFeatures=amountFeatures)
		print("Log written")

	elif(mode == enumerators.MODE.EXTREMESEARCH):
		createLogOfParametersExtremes(logfilename=logfilename, netID=netID, dataID=dataID, lossID=lossID, time=time, elapsedTime=elapsedTime, lossfunction=lossfunctionTesting, 
									numTraining=numTraining, numTotal=numTotal, insertedAoA=insertedAoa, extremesData=extremes, bestExtremes=bestExtremes, 
									foundcolareas=foundcolareas, foundrowareas=foundrowareas, width=width, newline=newline, separator1=separator1, separator2=separator2,
									searchInTraining=searchInTraining, randomVal=randomVal, usePrevRandomVal=usePrevRandomVal, shuffling=shuffling, amountFeatures=amountFeatures)
		print("Log written")

	elif(mode == enumerators.MODE.APPLYBASETODATA):
		createLogOfParametersBaseToData(logfilename=logfilename, dataID=dataID, baseID=baseID, time=time, elapsedTime=elapsedTime, numTotal=numTotal, width=width, newline=newline, 
									separator1=separator1, separator2=separator2, dataname=savedname,  directory=directory,  basedirectory=base_directory, roomName=roomName, 
									excluded=excluded, officeRows=officeRows, officeCols=officeCols, coordInfo=coordInfo, coordMode=coordMode, signalrepr=signalrepr, 
									samples=numTotal, samplesPerTarget=samplesPerTarget,baseNo=baseNo)
		print("Log written")
	elif(mode == enumerators.MODE.CDFPLOT or (mode == enumerators.MODE.ANALYZETUNEMODELS and analyzeCDFAndLog)):
		createLogOfParametersCDF(logfilename=logfilename, netID=netID, dataID=dataID, lossID=lossID, time=time, elapsedTime=elapsedTime, lossfunction=lossfunctionTesting, 
									numTraining=numTraining, numTotal=numTotal, insertedAoA=insertedAoa, meanError=meanError, width=width, newline=newline, separator1=separator1, separator2=separator2,
									usePrevRandomVal=usePrevRandomVal, shuffling=shuffling, randomVal=randomVal,
									filterDataAfterLoading=filterDataAfterLoading, mask=maskID, excluded=excluded, officeRows=officeRows, 
									officeCols=officeCols,useCombinedData=useCombinedData, baseNo=baseNo,amountFeatures=amountFeatures)
		print("Log written")
	elif(mode == enumerators.MODE.COAXLOADER):
		createLogOfParametersData(dataname=savedname, logfilename=logfilename, directory=coax_directory, dataID=baseID, time=time, 
									elapsedTime=elapsedTime, roomName=roomName, excluded=False, officeRows=enumerators.ROOM_ROW.COMPLETE, officeCols=enumerators.ROOM_COL.COMPLETE, 
									coordInfo=coaxInfo, coordMode=coordMode, signalrepr=signalrepr, 
									samples=numTotal, samplesPerTarget=coaxFilesPerAntenna, width=width, newline=newline, separator1=separator1, separator2=separator2,
									rmPilots=rmPilots, rmUnused=rmUnused, mask=maskID)
		print("Log written")
	elif(mode == enumerators.MODE.GETAOAOFDATA):
		createLogOfParametersGETAOA(dataname=savedname, logfilename=logfilename, dataID=dataID, time=time, elapsedTime=elapsedTime, roomName=roomName, 
									samples=numTotal,  width=width, newline=newline, separator1=separator1, separator2=separator2)
	else:
		print("No Log necessary")

#------------------------------------------ DATA Logging -------------------------------------------------------------#

def createLogOfParametersData(dataname, logfilename, directory, dataID, time, elapsedTime, roomName, excluded, officeRows, officeCols, 
	coordInfo, coordMode, signalrepr, samples, samplesPerTarget, width, newline, separator1, separator2 ,rmPilots, rmUnused, mask):
	
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-3)*" " + "DATA" + (width//2-3)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(-1, dataID, -1, -1, -1, -1, time, elapsedTime, newline, separator2, width))

	line = finishLine("| saved as %s" % (dataname), width)
	f.write(line + newline + separator2 + newline)

	samplesTotal = 0

	if(isinstance(directory, list)):
		for i in range(0,len(directory)):
			if(len(directory[i]) > 60):
				writedirectory = ".." + directory[i][-60:]
			line = finishLine("| directory %d: " % (i+1) +writedirectory, width)
			f.write(line + newline)
	else:
	# Print other parameters
	# Display only last 60chars of directory
		if(len(directory) > 60):
			writedirectory = ".." + directory[-60:]
		line = line = finishLine("| directory: "+writedirectory, width)
		f.write(line + newline + separator2 + newline)

	f.write(separator2 + newline)

	line = finishLine("| used room: " + roomName, width)
	f.write(line + newline)

	name = "excluded area: "
	if(excluded == False):
		name = "constrained area: "

	line = finishLine("| " + name + officeCols.name + " / " + officeRows.name, width)
	f.write(line + newline + separator2 + newline)

	if(isinstance(coordInfo[0], list)):
		for i in range(0,len(coordInfo)):
			line = finishLine("| coord. info %d: range from %03d to %03d with receiver id %2d and transmitter id %2d" % (i+1, coordInfo[i][0], coordInfo[i][1], coordInfo[i][2], coordInfo[i][3]), width)
			f.write(line + newline)
	else:
		line = finishLine("| coord. info: range from %03d to %03d with receiver id %2d and transmitter id %2d" % (coordInfo[0], coordInfo[1], coordInfo[2], coordInfo[3]), width)
		f.write(line + newline)

	line = finishLine("| coordinates format: " + coordMode.name, width)
	f.write(line + newline)

	line = finishLine("| signal representation: " + signalrepr.name, width)
	f.write(line + newline)		

	line = finishLine("| remove unused: %s; remove pilots: %s" % (rmUnused, rmPilots), width)
	f.write(line + newline + separator2 + newline)	

	if(mask != -1):
		line = finishLine("| Mask was applied: %s" % (mask), width)
		f.write(line + newline + separator2 + newline)

	if(isinstance(samplesPerTarget, list)):
		for i in range(0,len(samplesPerTarget)):
			line = finishLine("| Samples in %d: %d (%d per target)" % (i+1, samples[i], samplesPerTarget[i]), width)
			f.write(line + newline)
			samplesTotal = samplesTotal + samples[i]
		line = finishLine("| Amount of samples: %d" % (samplesTotal), width)
		f.write(line + newline + separator1 + newline + newline)	
	else:
		line = finishLine("| Amount of samples %d (%d samples per target)" % (samples, samplesPerTarget), width)
		f.write(line + newline + separator1 + newline + newline)	

	f.close()	

#----------------------------------------- BASE TO DATA Logging ----------------------------------------------------------#

def createLogOfParametersBaseToData(logfilename, dataID, baseID, time, elapsedTime, numTotal, width, newline, separator1, separator2, dataname,  
									directory,  basedirectory, roomName, excluded, officeRows, officeCols, coordInfo, coordMode, signalrepr, 
									samples, samplesPerTarget, baseNo):
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-11)*" " + "BASE APPLIED TO DATA" + (width//2-11)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(-1, baseID+"_"+dataID, -1, -1, -1, -1, time, elapsedTime, newline, separator2, width))

	line = finishLine("| saved as %s" % (dataname), width)
	f.write(line + newline + separator2 + newline)


	line = finishLine("| The data was combined with the sample No. %d" % (baseNo), width)
	f.write(line + newline + separator2 + newline)

	samplesTotal = 0

	if(isinstance(directory, list)):
		for i in range(0,len(directory)):
			if(len(directory[i]) > 60):
				writedirectory = ".." + directory[i][-60:]
			line = line = finishLine("| directory %d: " % (i+1) +writedirectory, width)
			f.write(line + newline)
	else:
	# Print other parameters
	# Display only last 60chars of directory
		if(len(directory) > 60):
			writedirectory = ".." + directory[-55:]
		line = line = finishLine("| data directory: "+writedirectory, width)
		f.write(line + newline + separator2 + newline)

	f.write(separator2 + newline)

	# Display only last 60chars of directory
	if(len(basedirectory) > 60):
		writedirectory = ".." + basedirectory[-51:]
	line = line = finishLine("| baseline directory: "+writedirectory, width)
	f.write(line + newline + separator2 + newline)

	f.write(separator2 + newline)

	line = finishLine("| used room: " + roomName, width)
	f.write(line + newline)

	name = "excluded area: "
	if(excluded == False):
		name = "constrained area: "

	line = finishLine("| " + name + officeCols.name + " / " + officeRows.name, width)
	f.write(line + newline + separator2 + newline)

	if(isinstance(coordInfo[0], list)):
		for i in range(0,len(coordInfo)):
			line = finishLine("| coord. info %d: range from %03d to %03d with receiver id %2d and transmitter id %2d" % (i+1, coordInfo[i][0], coordInfo[i][1], coordInfo[i][2], coordInfo[i][3]), width)
			f.write(line + newline)
	else:
		line = finishLine("| coord. info: range from %03d to %03d with receiver id %2d and transmitter id %2d" % (coordInfo[0], coordInfo[1], coordInfo[2], coordInfo[3]), width)
		f.write(line + newline)

	line = finishLine("| coordinates format: " + coordMode.name, width)
	f.write(line + newline)

	line = finishLine("| signal representation: " + signalrepr.name, width)
	f.write(line + newline + separator2 + newline)	

	if(isinstance(samplesPerTarget, list)):
		for i in range(0,len(samplesPerTarget)):
			line = finishLine("| Samples in %d: %d (%d per target)" % (i+1, samples[i], samplesPerTarget[i]), width)
			f.write(line + newline)
			samplesTotal = samplesTotal + samples[i]
		line = finishLine("| Amount of samples: %d" % (samplesTotal), width)
		f.write(line + newline + separator1 + newline + newline)	
	else:
		line = finishLine("| Amount of samples %d (%d samples per target)" % (samples, samplesPerTarget), width)
		f.write(line + newline + separator1 + newline + newline)	

	f.close()	

# ------------------------------------------ GET AOA OF DATA ---------------------------------------------------------#

def createLogOfParametersGETAOA(dataname, logfilename, dataID, time, elapsedTime, roomName, 
									samples,  width, newline, separator1, separator2):
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-8)*" " + "GET AOA OF DATA" + (width//2-9)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(-1, dataID, -1, -1, -1, -1, time, elapsedTime, newline, separator2, width))

	line = finishLine("| saved as %s" % (dataname), width)
	f.write(line + newline + separator2 + newline)

	line = finishLine("| used room: " + roomName, width)
	f.write(line + newline)

	line = finishLine("| Amount of samples %d" % (samples), width)
	f.write(line + newline + separator1 + newline + newline)	

	f.close()	

#----------------------------------------- TRAINING Logging ----------------------------------------------------------#

def createLogOfParametersTraining(netname, logfilename, netID, dataID, lossID, time, elapsedTime, learningrate, momentum, batchsize, updatefreq,
	droprate, kernelsize, inchannels, epochsTotal, lossfunctionTraining, lossfunctionTesting, numTraining, numTotal, insertedAoA, meanError, backprop, 
	width, newline, separator1, separator2, dropoutEnabled, shuffling, selectFromSamplesPerPos, normalize, usePrevRandomVal,
	filterDataAfterLoading, randomVal, mask, excluded, officeRows, officeCols, useCombinedData, baseNo, amountFeatures):
	
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-5)*" " + "TRAINING" + (width//2-5)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width))

	line = finishLine("| saved as %s" % (netname), width)
	f.write(line + newline + separator2 + newline)


	if(useCombinedData):
		line = finishLine("| The data was combined with the sample No. %d" % (baseNo), width)
		f.write(line + newline + separator2 + newline)

	if(mask != -1):
		line = finishLine("| Mask was applied: %s" % (mask), width)
		f.write(line + newline)

	line = finishLine("| Selected from %d samples per position" % (selectFromSamplesPerPos), width)
	f.write(line + newline)

	line = finishLine("| Data was filtered after Loading: %s" % (filterDataAfterLoading), width)
	f.write(line + newline + separator2 + newline)

	if(filterDataAfterLoading):
		name = "excluded area: "
		if(excluded == False):
			name = "constrained area: "

		line = finishLine("| " + name + officeCols.name + " / " + officeRows.name, width)
		f.write(line + newline + separator2 + newline)

	# Print other parameters
	line = finishLine("| AoA used as extra feature: %s" % (insertedAoA), width) + newline

	line = line + finishLine("| Hidden units in the fully connected layer: %d" % (int(amountFeatures)), width) + newline

	f.write(line)

	line = finishLine("| Normalization: %s" % (normalize), width)
	f.write(line + newline + separator2 + newline)

	line = finishLine("| backprop algorithm: %s" % (backprop.name), width)
	f.write(line + newline)

	line = finishLine("| learningrate: %.04f, momentum: %.1f, batchsize: %d, learninigUpdateFreq: %d" % (learningrate, momentum, batchsize, updatefreq), width)
	f.write(line + newline)

	line = finishLine("| droprate: %.1f (Enabled: %s)" % (droprate, dropoutEnabled), width)
	f.write(line + newline)

	line = finishLine("| kernelsize: %d, inchannels: %d" % (kernelsize, inchannels), width)
	f.write(line + newline)

	line = finishLine("| epochs: %d" % (epochsTotal), width)
	f.write(line + newline + separator2 + newline)

	line = finishLine("| lossfunction used for training: " + lossfunctionTraining.name, width)
	f.write(line + newline)

	line = finishLine("| lossfunction used for testing: " + lossfunctionTesting.name, width)
	f.write(line + newline)

	line = finishLine("| amount of samples used for training: %d (of %d total samples)" % (numTraining, numTotal), width)
	f.write(line + newline + separator2 + newline)

	# Print results
	line = finishLine("| Achieved mean training error: %.3f" % (meanError), width)
	f.write(line + newline + separator1 + newline + newline)	

	f.close()

#----------------------------------------- TESTING Logging -----------------------------------------------------------#


def createLogOfParametersTesting(logfilename, netID, dataID, lossID, time, elapsedTime, lossfunction, numTraining, numTotal, insertedAoA, meanError, 
								width, newline, separator1, separator2, randomVal, usePrevRandomVal, shuffling, mask, baseNo, useCombinedData,
								filterDataAfterLoading, excluded, officeRows, officeCols, normalize, amountFeatures):
	
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-5)*" " + "TESTING" + (width//2-4)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width))

	if(useCombinedData):
		line = finishLine("| The data was combined with the sample No. %d" % (baseNo), width)
		f.write(line + newline + separator2 + newline)

	if(mask != -1):
		line = finishLine("| Mask was applied: %s" % (mask), width)
		f.write(line + newline)

	line = finishLine("| Data was filtered after Loading: %s" % (filterDataAfterLoading), width)
	f.write(line + newline)

	if(filterDataAfterLoading):
		name = "excluded area: "
		if(excluded == False):
			name = "constrained area: "

		line = finishLine("| " + name + officeCols.name + " / " + officeRows.name, width)
		f.write(line + newline + separator2 + newline)

	# Print other parameters
	line = finishLine("| AoA used as extra feature: %s" % (insertedAoA), width) + newline

	line = line + finishLine("| Hidden units in the fully connected layer: %d" % (int(amountFeatures)), width) + newline

	f.write(line)
	line = finishLine("| Normalization: %s" % (normalize), width)
	f.write(line + newline)

	line = finishLine("| lossfunction: " + lossfunction.name, width)
	f.write(line + newline)

	line = finishLine("| amount of test samples: %d (of %d total samples)" % (numTotal, numTotal+numTraining), width)
	f.write(line + newline + separator2 + newline)

	# Print results
	line = finishLine("| Achieved mean testing error: %.3f" % (meanError), width)
	f.write(line + newline + separator1 + newline + newline)	

	f.close()

# --------------------------------------------------- CDF Logging --------------------------------------------------------- #

def createLogOfParametersCDF(logfilename, netID, dataID, lossID, time, elapsedTime, lossfunction, numTraining, numTotal, insertedAoA, meanError, 
								width, newline, separator1, separator2, usePrevRandomVal, shuffling, randomVal, filterDataAfterLoading, mask, excluded,
								 officeRows, officeCols,useCombinedData, baseNo, amountFeatures):
	
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")
	
	# Print headline
	line = "|" + (width//2-3)*" " + "CDF" + (width//2-2)*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width))

	if(useCombinedData):
		line = finishLine("| The data was combined with the sample No. %d" % (baseNo), width)
		f.write(line + newline + separator2 + newline)

	if(mask != -1):
		line = finishLine("| Mask was applied: %s" % (mask), width)
		f.write(line + newline)

	line = finishLine("| Data was filtered after Loading: %s" % (filterDataAfterLoading), width)
	f.write(line + newline)

	if(filterDataAfterLoading):
		name = "excluded area: "
		if(excluded == False):
			name = "constrained area: "

		line = finishLine("| " + name + officeCols.name + " / " + officeRows.name, width)
		f.write(line + newline + separator2 + newline)

	# Print other parameters
	line = finishLine("| AoA used as extra feature: %s" % (insertedAoA), width) + newline

	line = line + finishLine("| Hidden units in the fully connected layer: %d" % (int(amountFeatures)), width) + newline

	f.write(line)

	line = finishLine("| lossfunction: " + lossfunction.name, width)
	f.write(line + newline)

	line = finishLine("| amount of test samples: %d (of %d total samples)" % (numTotal, numTotal+numTraining), width)
	f.write(line + newline + separator2 + newline)

	# Print results
	line = finishLine("| Achieved first quartile:  %.3f" % (meanError[0]), width)
	f.write(line + newline)
	line = finishLine("| Achieved median error  :  %.3f" % (meanError[1]), width)
	f.write(line + newline)
	line = finishLine("| Achieved third quartile:  %.3f" % (meanError[2]), width)
	f.write(line + newline + separator1 + newline + newline)	

	f.close()

# ---------------------------------------- EXTREMES Logging ----------------------------------------------------------- #

def createLogOfParametersExtremes(logfilename, netID, dataID, lossID, time, elapsedTime, lossfunction, numTraining, numTotal, insertedAoA,
	extremesData, bestExtremes, foundcolareas, foundrowareas, width, newline, separator1, separator2, searchInTraining, randomVal, usePrevRandomVal, 
	shuffling, amountFeatures):
	
	# Open File with "a" => append a log entry to the logfile
	f = open(logfilename, "a")

	
	# Split data in extremes and estimated range
	extremes = extremesData[0]
	estimated_range = extremesData[1]
	
	# Print headline
	extremesLen = str(len(extremes))

	if(bestExtremes):
		line = "|" + (width//2-6)*" " + "BEST " + extremesLen + (width//2-1-len(extremesLen))*" " + "|"
	else:
		line = "|" + (width//2-6)*" " + "WORST " + extremesLen + (width//2-2-len(extremesLen))*" " + "|"
	f.write(newline + separator1 + newline + line + newline + separator2 + newline)

	# Print header
	f.write(printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width))

	if(searchInTraining):
		line = finishLine("| Searched in TrainingSet!", width)
	else:
		line = finishLine("| Searched in the Test Set", width)

	f.write(line + newline + separator2 + newline)

	# Print other parameters
	line = finishLine("| AoA used as extra feature: %s" % (insertedAoA), width) + newline

	line = line + finishLine("| Hidden units in the fully connected layer: %d" % (int(amountFeatures)), width) + newline

	f.write(line)

	line = finishLine("| lossfunction: " + lossfunction.name, width)
	f.write(line + newline)

	line = finishLine("| amount of test samples: %d (of %d total samples)" % (numTotal, numTotal+numTraining), width)
	f.write(line + newline + separator2 + newline)

	# Print results
	line = finishLine("| indices found in column areas: %s" % (foundcolareas), width)
	f.write(line + newline)	

	line = finishLine("| indices found in row areas: %s" % (foundrowareas), width)
	f.write(line + newline + separator2 + newline)	

	line = finishLine("| estimated indices in the range of (%.3f - %.3f | %.3f - %.3f)" % (estimated_range[0][0], estimated_range[0][1], estimated_range[1][0], estimated_range[1][1]), width)
	f.write(line + newline + separator2 + newline)	


	if(bestExtremes):
		line = finishLine("| best errors in the range from %.3f to %.3f" % (extremes[0][0], extremes[len(extremes)-1][0]), width)
	else:
		line = finishLine("| worst errors in the range from %.3f to %.3f" % (extremes[0][0], extremes[len(extremes)-1][0]), width)

	f.write(line + newline + separator1+ newline)			

	f.close()

# ########################################## LOG RAY TUNE RES ########################################## #

def getLogString(netname, logfilename, netID, dataID, lossID, time, elapsedTime, learningrate, momentum, batchsize,
	droprate, kernelsize, inchannels, epochsTotal, lossfunctionTraining, lossfunctionTesting, numTraining, epochSize, testSize, insertedAoA, meanError, backprop, 
	width, dropoutEnabled, shuffling, selectFromSamplesPerPos, normalize, usePrevRandomVal,
	filterDataAfterLoading, randomVal, mask, excluded, officeRows, officeCols, useCombinedData, baseNo, amountFeatures, numValidation):
	
	newline = "\n"
	separator1 = "+" + "="*(width-2) + "+"
	separator2 = "+" + "-"*(width-2) + "+"

	# Print headline
	line = "|" + (width//2-5)*" " + "TRAINING" + (width//2-5)*" " + "|"
	line = newline + separator1 + newline + line + newline + separator2 + newline

	# Print header
	line = line + printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width)

	line = line + finishLine("| saved as %s" % (netname), width) + newline + separator2 + newline

	if(useCombinedData):
		line = line + finishLine("| The data was combined with the sample No. %d" % (baseNo), width) + newline + separator2 + newline

	if(mask != -1):
		line = line + finishLine("| Mask was applied: %s" % (mask), width) + newline

	line = line + finishLine("| Selected from %d samples per position" % (selectFromSamplesPerPos), width) + newline

	line = line + finishLine("| Data was filtered after Loading: %s" % (filterDataAfterLoading), width) + newline + separator2 + newline

	if(filterDataAfterLoading):
		name = "excluded area: "
		if(excluded == False):
			name = "constrained area: "

		line = line + finishLine("| " + name + officeCols.name + " / " + officeRows.name, width) + newline + separator2 + newline

	# Print other parameters
	line = line + finishLine("| AoA used as extra feature: %s" % (insertedAoA), width) + newline

	line = line + finishLine("| Hidden units in the fully connected layer: %d" % (int(amountFeatures)), width) + newline

	line = line + finishLine("| Normalization: %s" % (normalize), width) + newline + separator2 + newline

	line = line + finishLine("| backprop algorithm: %s" % (backprop.name), width) + newline

	line = line + finishLine("| learningrate: %.04f, momentum: %.1f, batchsize: %d" % (learningrate, momentum, batchsize), width) + newline

	line = line + finishLine("| droprate: %.1f (Enabled: %s)" % (droprate, dropoutEnabled), width) + newline

	line = line + finishLine("| kernelsize: %d, inchannels: %d" % (kernelsize, inchannels), width) + newline

	line = line + finishLine("| epochs: %d" % (epochsTotal), width) + newline + separator2 + newline

	line = line + finishLine("| lossfunction used for training: " + lossfunctionTraining.name, width) + newline

	line = line + finishLine("| lossfunction used for testing: " + lossfunctionTesting.name, width) + newline

	line = line + finishLine("| Epochsize: %d; Testsize: %d" % (epochSize, testSize),width) + newline

	line = line + finishLine("| amount of samples used for training: %d" % (numTraining), width) + newline

	line = line + finishLine("| amount of samples used for validation: %d" % (numValidation), width) + newline + separator2 + newline

	# Print results
	line = line + finishLine("| Achieved mean testing error: %.3f" % (meanError), width) + newline + separator1 + newline + newline

	return line

#---------------------------------------- Helper method Logging ------------------------------------------------------#

# Print Header consisting of: data id, netid (if necessary), executed time and elapsed time
def printHeader(netID, dataID, lossID, randomVal, usePrevRandomVal, shuffling, time, elapsedTime, newline, separator2, width):
	
	line = ""
	if(netID == -1):
		line = finishLine("| dataID: %s" % (dataID), width)
	else:
		line = finishLine("| netID: %s lossID: %s dataID: %s" % (netID, lossID, dataID), width)

	line = line + newline + separator2 + newline

	if shuffling != -1:
		line = line + finishLine("| Shuffling activated: %s" % (shuffling), width) + newline

		line = line + finishLine("| Random permutation: %s (Use previous: %s)" % (randomVal, usePrevRandomVal), width)

		line = line + newline + separator2 + newline

	line = line + finishLine("| Executed: " + time, width) + newline

	line = line + finishLine("| Elapsed time: " + convertTimeFormat(elapsedTime) + " (HH:MM:SS.mmm)", width)
	line = line + newline + separator2 + newline

	return line

# Finishline => add whitespaces and terminate the line with '|'
def finishLine(text, width):
	l = len(text)
	line = text + " "*(width-l-1) + "|"
	return line

###########################################################################################################################