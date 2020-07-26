#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Kosterhon

This class defines all necessary parameters to use the framework for training, testing, visualization etc.
"""

######################       IMPORTANT:      ##################################    
#
# The directory for all the data has to be defined in dataParameters.py !!!
#
############################################################################### 

# ------------------------------------------- IMPORTS -------------------------------------------------------------- #

import roomParameters
import dataParameters
import enumerators

import numpy as np

############################################# Model IDs ##############################################################

# Only used for ANALYZETUNEMODELS Mode !

modelIDs_ExampleRun = ["model_2020-07-2613-26-509p43pzyj", "model_2020-07-2613-27-16xsuej8qr", "model_2020-07-2613-27-34", "model_2020-07-2613-28-08ebbhxf", "model_2020-07-2613-28-199vs2f29a"]

############################################## All Parameters ########################################################
######################################################################################################################

# -------------------------------------------------- MODES / ModelIDs ---------------------------------------------- #

# Activate automatedMode
# Should be activated, if one or multiple of the following parameters are
# adapted during program execution -> adaption done in main.py: updateParameters(index) dependent on iteration index
automatedMode = False

# Possible Modes #
################## 
#	DATALOADER, BASELOADER, COAXLOADER, 
#   APPLYMASK, APPLYBASETODATA,GETAOAOFDATA,
# 	TRAINING, TESTING, EXTREMESEARCH, ANALYZETUNEMODELS,
# 	VISUALIZATION, CDFPLOT, CREATEHEATMAP, CREATEHITMAP,
# 	REFINEDATALOADER, REFINENETWORK
#   INSTANT, USER_DEFINED 

# Example modes:
# modes = [enumerators.MODE.DATALOADER]
# modes = [enumerators.MODE.TRAINING, enumerators.MODE.TESTING, enumerators.MODE.CDFPLOT]
# modes = [enumerators.MODE.EXTREMESEARCH, enumerators.MODE.VISUALIZATION]
# modes = [enumerators.MODE.REFINEDATALOADER, enumerators.MODE.REFINENETWORK]

modes = [enumerators.MODE.TRAINING, enumerators.MODE.TESTING, enumerators.MODE.CDFPLOT]

# --------------------------------------------------------- ID s --------------------------------------------------- #

# ID to identify the data
dataID = "exampleID"
saved_dataID = dataID

# Refine ID
refineDataID = "exampleRefinement"
refineDataIDSaved = refineDataID

# B__ -> Empty room as baseline
# C__ -> Coax Cable measurement
# Also specify, which sample should be used
baseID = "B01"
baseNo = 10

# Is used -> dataID = dataID + baseID -> load combined data
useCombinedData = True

# Convention so far: N = New Pipeline, A = AoA used;
netID = "NA01"

# If string is not empty -> load this instead of a network!
loadModelAsNet = ""

# Load refined network
loadRefinedNetwork = False

# Loss ID => Use it to quickly distinguish betw networks trained with different loss functions
lossID = "L001"

# ---------------------------------------------- GENERAL DIRECTORIES ---------------------------------------------- #

# LogFile for the Tune Training / optimization
tuneLogFile 	= "/Path/To/sampleFiles/logs/tuneResults_Example.log"

# Directory to store the network, random_ids etc.
storeDirectory 	= "/Path/To/sampleFiles/output/"

# -------------------------------------------- DATA PROCESSING Parameters ---------------------------------------- #

####### Filtering while loading the data via DATALOADER (Preprocessing) #######

# Choose type of coordinates and signalrepresentation
coordMode = enumerators.COORDINATES.CARTESIAN

# IMPORTANT: ONLY IQ SHOULD BE CHOOSEN 
# -> some modes like AoA only work for IQ!! (also the normalization!)
signalrepr = enumerators.SIGNALREPRESENTATIONS.IQ

# Remove pilots and unused stuff - should be set to True
rmPilots = True
rmUnused = True

#How many samples to analyze for pilot tones
samplesToAnalyze = 100 

##################### Filtering after loading the data ###################### 

# Uses all the data instead of only the testdata for testing
# This makes sense, when e.g. the data is loaded from another day than the training
testAllData = False 

# Load refineData for testing
loadRefineDataAsTest = False

# NORMALIZE
normalizeData = False

# Select which antennas to use
# Keep in mind, that AoA should not be used with less antennas!
antennasRXUsed = {1,2,3,4}
antennasTXUsed = {1,2,3,4}

antennasRX = len(antennasRXUsed)
antennasTX = len(antennasTXUsed)

# If pilots and unused removed -> 234
# If the value is less than 234 => Constrain amount of subcarriers
samplesPerChannel = 234

# Always <= filesPerTarget
selectFromSamplesPerPos = 12

# If mask != -1 --> Apply mask to filter the data for the training
# Can be found in roomParameters!
# e.g. mask = roomParameters.Mask_Checkerboard1 & maskID = "Checkerboard_1"
mask = -1 #roomParameters.BehindWallMask #-1 
maskID = -1 #"BehindWallMask" #-1

# If ALL the Data is requested (e.g. AoA) -> No shuffling as it makes no sense
shuffling = True

# Use previous random permutation (if used, randomDigest contains the ID/ Hash)
usePrevRandomVal = False
randomDigest = ""

# Use the ROOM Parameter constraints (exceptRow, exceptCol, excluded)
# Room constraints defined below in own section
filterDataAfterLoading = False

# ---------------------------------------- LEARNING Params ------------------------------------------- #

# Total epochs
epochsTotal = 40	

# Amount of trainingsdata
numTraining = 100

# Amount of data for validation
numValidation = 100

# Batchsize
batchsize = 5

# Use prev. calculated AoA as a input-feature
insertAoAInNetwork = True

# Amount of hidden unit in the classification layer
amountFeatures = 150

# Amount of Trainngsamples for Refinement (12er: 1920 insg.)
refinementNumTraining = 1000

# Valid for IQ Samples
inchannels = antennasRX * antennasTX * 2

# Backpropagation algorithm
backprop = enumerators.BACKPROP.ADAM

# Initial learningrate 
learningrate = 0.001

# Only used with SGD
momentum = 0.9  	#only used with SGD
updatefreq = 10  	# only used with SGD; update learningrate

# kernelsize of Conv. Layer
kernelsize = 5

# Dropout, Rate only matters if enabled
dropoutEnabled = False
droprate = 0.3

# Lossfunctions used for Training and Testing
lossfunctionTesting = enumerators.LOSSFUNCTIONS.CARTESIAN_DIST

lossfunctionTraining = enumerators.LOSSFUNCTIONS.CARTESIAN_DIST 

# ---------------------------------- Extreme Search / Visualize Parameters --------------------------------------- #

# Extremesearch params
numExtremes=50
bestExtremes=True

# Visualize: If not random data -> load prev. calculated extreme data
visualizeRandomData = False    

# For extremesearch & visualize: search/visualize in Test- or Trainingsset
searchInTraining = False

# ----------------------------------------------- AOA Parameters ------------------------------------------------- #

# If activated get AoA of Refinement Data
getAoARefinement = False

# --------------------------------------------------- HEATMAP / HITMAP ---------------------------------------------------- #

# If used, all hits are shown instead of for each location
visualizeAllDataHits = False

# Should be set automatically when using automatedMode! -> -1 = not_used / infinite
heatMapTimeout = -1

# If the mode is automated, set timeout to xx seconds
if(automatedMode):
	heatMapTimeout = 2

# ----------------------------------------------- TUNE ANALYSIS --------------------------------------------------- #

# directory where all models trained by tune are stored:
modelDir = "exampleRun/"

# Define the modelIDs ()
modelIDs = modelIDs_ExampleRun
modelID = modelIDs[0]

# Uncomment following line to analyze all given models
modes = [enumerators.MODE.ANALYZETUNEMODELS] * len(modelIDs)

analyzeCDFAndLog = True

# ------------------------------------------------ INSTANT MODE -------------------------------------------------- #

# include VideoMode parameter for instant mode
# if videoMode is activated, the room is rotated 90 degrees to display it better in demonstration videos (same rotation as the room)
videoMode = False

# Set Timeout for showing the position graphically on the grid
timeout_Instant = 5
packetsToCombine = 1
averageInput = False
 
# -------------------------------------------- Logging and Debugging --------------------------------------------- #

# LogFile for the evaluation / own methods
logFile = "sampleFiles/logs/example.log"

# Print on console each xxx during the training
printfreq = 10

# Logging parameters for the log
logWidth = 80
loggingEnabled = True

# Debug process -> many plots / visualizations (have to be closed manually!)
debug = False

####################################################################################################################
####################################################################################################################

# ---------------------------------------------- EXPERIMENT Parameters ------------------------------------------- #

# Sending parameters
antennadist = 0.02; 
frequency = 5.785 * (10**9); #GHz
wavelength = (3e8) / frequency

# Usable: +-2 until +- 122 without pilots (RANGE FROM -128 to 127 => Subcarrier 0 existent but useless! (Index 128 => 0)
# Reference: https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html
pilots = np.array([11, 39, 75, 103])
datarange = np.array([2, 122])

# Which kernel + CSI extractor is used
importMethod = enumerators.IQCONVERTER.NEW

# --------------------------------------------------- ROOM Parameters -------------------------------------------- #

# defined in roomParameters.py!

# SELECT AREA OF ROOM: Currently -> complete Room, all columns + all rows
exceptRow = enumerators.ROOM_ROW.COMPLETE	
exceptCol = enumerators.ROOM_COL.COMPLETE
excluded = False    # Alternative if excluded == False: Constrained setting -> only take what is in there!

# Determine room and roomName 
# Room = assign measurements to positions in the grid
room = roomParameters.Classroom_12er
roomName = "Example Scenario"

# Room for the refinement data
roomRefinement = roomParameters.Classroom_12er

# Determine baseline and coaxSetup 
baseline = roomParameters.Baseline
coaxSetup = roomParameters.CoaxAntennas

# Assign the rows and cols to the rows/cols of the room
rows = roomParameters.ClassroomRows
cols = roomParameters.ClassroomCols

# --------------------------------------------------- DATA Parameters -------------------------------------------- #

# Data Parameters
coordInfo = dataParameters.coordInfo_ExampleRoom_2507
filesPerTarget = dataParameters.filesPerTarget_ExampleRoom_2507
directory = dataParameters.directory_ExampleRoom_2507

# Base Parameters (B__)
base_config = dataParameters.baselineInfo
base_files = dataParameters.filesBaseline_ExampleRoom_2507
base_directory = dataParameters.basedirectory_ExampleRoom_2507

# Coax Parameters (C__)
coax_config = dataParameters.coaxInfo
coaxFilesPerAntenna = dataParameters.filesCoaxPerAntenna
coax_directory = dataParameters.coaxdirectory

# Data Parameters for Refinement
coordInfoRefinement = dataParameters.coordInfo_ExampleRoom_2507
filesPerTargetRefinement = dataParameters.filesPerTarget_ExampleRoom_2507
directoryRefinement = dataParameters.directory_ExampleRoom_2507

####################################################################################################################
####################################################################################################################

# DO NOT CHANGE THIS!

# Adapt dataID if necessary
if(useCombinedData and not enumerators.MODE.APPLYBASETODATA in modes):
    dataID = baseID+"_"+dataID

# Calculate the exceptRoom
roomCompl = roomParameters.ClassroomComplete

exceptColData = cols[exceptCol.value]
exceptRowData = rows[exceptRow.value]

# The specified columns and rows are now added to the area, which they describe -> the intersection of both! (not union)
if(exceptRowData == roomParameters.Empty):
    excludedArea = set()
else:
    excludedArea = exceptRowData.intersection(exceptColData)

# Not excluded -> Area constrained to what is given -> invert it, as it is passed for exclusion of values
if not excluded:
    excludedArea = roomCompl.difference(excludedArea)

####################################################################################################################
####################################################################################################################