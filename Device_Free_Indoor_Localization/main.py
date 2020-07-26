#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

This is the main class, which is called from the terminal
Depending on the parameter configuration in parameters.py, the corresponding functionality is provided

"""

############################################## IMPORTS #############################################################

from torch.utils.data import Dataset
from datetime import datetime
from colour import Color

import os
import time
import sys
import torch
import random
import pyttsx3
import numpy as np
from colored import fg, bg, attr

import enumerators
import utils
import visualize
import plottingUtils
import learning
import dataManagement
import AoA
import essentialMethods

import parameters

#####################################################################################################################################

# Parameters to change in updateParameters(index)
# Just an example 

insertAoArray= [False,True]

##################################### Parameter adaption #############################################################################

# Update Parameters for the automated Mode
def updateParameters(index):

    # Change if AoA is inserted for every iteration
    # This is just an example
    parameters.insertAoAInNetwork = insertAoArray[index % 2]
    utils.printInfo("Changed: AoA inserted: %s" % (parameters.insertAoAInNetwork))

#################################### Helper Method ##################################################################################

# Create a string, which shows the contained strings in a set concatenated with "," and "and"
def getSetContent(workSet, name):
    counter = 0
    print("Indices were found in the following %s sets: " % (name))
    foundareas = ""

    for index in workSet:
        counter = counter + 1
        print(index.name)
        
        if counter == 1:
            foundareas = index.name
        elif counter == len(workSet):
            foundareas = foundareas + " and " + index.name
        else:
            foundareas = foundareas + ", " + index.name

    return foundareas


######################################### Different Modes ##########################################################################

# -------------------------------------------------------------------------------------------------------------------------------- #

# Data Processing chain
def dataProcessing(errorLog=[]):

    # Get the data as well as params (length of one (or multiple directionaries))
    data, params, errorLog = dataManagement.getDataFromRawPipeline(directory=parameters.directory, coordInfo=parameters.coordInfo, room=parameters.room, coordMode=parameters.coordMode,
                      filesPerTarget=parameters.filesPerTarget, signalrepr=parameters.signalrepr, exception=parameters.excludedArea, 
                      pilots=parameters.pilots, datarange=parameters.datarange, debug=parameters.debug, rmPilots=parameters.rmPilots, rmUnused=parameters.rmUnused,
                      samplesToAnalyze=parameters.samplesToAnalyze, converter=parameters.importMethod, inchannels=parameters.inchannels, errorLog=errorLog)

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.dataID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # Print it if debug activated
    if(parameters.debug):
        print(filename)

    # Save data
    utils.saveFile(filename,data)

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, params, -1, -1, -1, -1, errorLog]


# -------------------------------------------------------------------------------------------------------------------------------- #

# Data Processing chain
def baselineProcessing(errorLog=[]):

    data, params, errorLog = dataManagement.getDataFromRawPipeline(directory=parameters.base_directory, coordInfo=parameters.base_config, room=parameters.baseline, coordMode=parameters.coordMode,
                      filesPerTarget=parameters.base_files, signalrepr=parameters.signalrepr, exception=set(), 
                      pilots=parameters.pilots, datarange=parameters.datarange, debug=parameters.debug, rmPilots=parameters.rmPilots, rmUnused=parameters.rmUnused,
                      samplesToAnalyze=parameters.base_files, converter=parameters.importMethod, inchannels=parameters.inchannels, errorLog=errorLog)

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.baseID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    if(parameters.debug):
        print(filename)

    # Save data
    utils.saveFile(filename,data)

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, params, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Load the measurements obtained after connecting the routers with coaxial cables
def loadCoax(errorLog=[]):

    # Get the data from the hdf5 files
    data, params, errorLog = dataManagement.getDataFromRawPipeline(directory=parameters.coax_directory, coordInfo=parameters.coax_config, room=parameters.coaxSetup, coordMode=parameters.coordMode,
                      filesPerTarget=parameters.coaxFilesPerAntenna, signalrepr=parameters.signalrepr, exception=set(), 
                      pilots=parameters.pilots, datarange=parameters.datarange, debug=parameters.debug, rmPilots=parameters.rmPilots, rmUnused=parameters.rmUnused,
                      samplesToAnalyze=100, converter=parameters.importMethod, inchannels=parameters.inchannels, errorLog=errorLog)

    filename = utils.createFileName(dataID=parameters.baseID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # The TX streams belong to the following antennas:
    # FrontLeft | FrontMiddle | FrontRight | Internal
    # TX 2      | TX 4        | TX 1       | TX 3
   
    data_Combined = []

    # Sanity check
    if(parameters.coaxFilesPerAntenna != len(data)//4):
        utils.printFailure("Error happened in loadCoax - Wrong Files Per Target relationship to data length")
        errorLog.append("Error happened in loadCoax - Wrong Files Per Target relationship to data length")

    # Assign the correct streams to the corresponding position
    # This combines the different measurements, as each time only one transmission antenna can be connected to all receiving antennas
    for d in range(0,len(data)//4):
        dataobj = torch.zeros(data[d][0].shape)
        label = data[d][1]

        dataobj[0:8,:] = data[d+2*parameters.coaxFilesPerAntenna][0][0:8,:] # Front Right - Tx 1
        dataobj[8:16,:] = data[d][0][8:16,:] # Front Left - Tx 2
        dataobj[16:24,:] = data[d+3*parameters.coaxFilesPerAntenna][0][16:24,:] # Internal - Tx 3
        dataobj[24:32,:] = data[d+1*parameters.coaxFilesPerAntenna][0][24:32,:] # Front Middle - Tx 4

        data_Combined.append([dataobj, label])

    if(parameters.debug):
        print(filename)

    # Save data
    utils.saveFile(filename,data_Combined)

    return [filename, params, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Filter the data according to a specified mask. Afterwards the filtered data is stored.
def applyMask(errorLog=[]):

    # Load data
    data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True, errorLog)

    # Create filename with the given parameters (-1 = not present/not necessary)
    # Append MaskID to filename
    filename = utils.createFileName(dataID=parameters.dataID+"_"+parameters.maskID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # Save data
    utils.saveFile(filename,data)

    # If AoA features are used: also store it in a separate file
    if parameters.insertAoAInNetwork:
        data_AoA = []

        # append the AoA data for the mask data to an array
        for i in range(0,len(data)):
            data_AoA.append(data[i][2])
        
        # store it as separate file
        filename = utils.createFileName(dataID=parameters.dataID+"_"+parameters.maskID+"_AoA", netID=-1, lossID=-1, extremes=-1, numExtremes=-1)
        utils.saveFile(filename, data_AoA)

    return [-1, -1, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Apply a given baseline to the data: either an empty room or a coaxial cable measurement
def applyBaseLineToData(errorLog=[]):

    # Store the original dataID (not combined one)
    parameters.dataID = parameters.saved_dataID

    # Load the data
    data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, False, errorLog)

    # Load the baseline
    base, baseLen = essentialMethods.loadBase()

    # Use one sample of the baseline as reference to normalize all data samples
    reference = base[parameters.baseNo][0]

    # Normalize every sample with the reference
    for i in range(0,dataLen):
        for k in range(0,len(reference)//2):

            # data as well as reference are converted to complex values (IQ samples) to normalize correctly
            # For each of the antenna-streams
            ref = np.array([np.add(reference[2*k].numpy(), np.multiply(reference[2*k+1].numpy(), 1j))], dtype=complex)
            dataSig = np.array([np.add(data[i][0][2*k].numpy(), np.multiply(data[i][0][2*k+1].numpy(), 1j))], dtype=complex)
            normalizedData = dataSig / ref

            # Store the normalized values
            data[i][0][2*k] = torch.from_numpy(np.real(normalizedData))
            data[i][0][2*k+1] = torch.from_numpy(np.imag(normalizedData))

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.baseID+"_"+parameters.dataID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    if(parameters.debug):
        print(filename)

    # Save data
    utils.saveFile(filename,data)

    # For further steps
    parameters.dataID = parameters.baseID+"_"+parameters.dataID

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, dataLen, -1, -1, -1, -1, errorLog]


# -------------------------------------------------------------------------------------------------------------------------------- #

# Calculate the Angle of Arrival (AoA) for the given Data (data or refinement)
def getAoAOfData(errorLog=[]):

    # Load the requested data (determined through parameters.py) and the corresponding amount of files per Target / position
    if parameters.getAoARefinement:
        data,dataLen, errorLog = essentialMethods.loadRefinementData(False, errorLog)
        filesPerTarget = parameters.filesPerTargetRefinement
    else:
        data,dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, False, errorLog)
        filesPerTarget = parameters.filesPerTarget

    # It if is a list -> data of multiple days = multiple directories
    if (isinstance(filesPerTarget,list)):

        dataAoA = []

        # !! IMPORTANT !!
        # Has to be changed manually right now, as the import of multiple files is not fully supported right now
        data = [data[0:12*filesPerTarget[0]], data[12*filesPerTarget[0]:12*filesPerTarget[0]+ 12*filesPerTarget[1]], data[12*filesPerTarget[0]+ 12*filesPerTarget[1]:12*filesPerTarget[0]+ 12*filesPerTarget[1]+12*filesPerTarget[2]], data[12*filesPerTarget[0]+ 12*filesPerTarget[1]+12*filesPerTarget[2]:]]
        
        # for each of the data directories, calculate the AoA
        for setindex in range(0,len(filesPerTarget)):

            for d in range(0,len(data[setindex])):

                dataobj = np.zeros([parameters.antennasTX, parameters.antennasRX, data[setindex][d][0].shape[1]], dtype=complex)

                # Create complex values
                for tx in range(0,parameters.antennasTX):
                    for rx in range(0,parameters.antennasRX):
                        for k in range(0,data[setindex][d][0].shape[1]):
                            dataobj[tx][rx][k] = data[setindex][d][0][2*(tx*parameters.antennasRX+rx)][k].item() + data[setindex][d][0][2*(tx*parameters.antennasRX+rx)+1][k].item() * 1j

                # Get the angles for all streams
                angle = AoA.applyMUSIC(dataobj, parameters.antennadist, parameters.wavelength, False, False)

                # Only consider the angles of three streams (instead of four)
                # The stream of the internal antenna is discarded!
                dataAoA.append([angle[0], angle[1], angle[3]])

    else:

        # Array to store the calculated angles
        dataAoA = []

        for d in range(0,len(data)):
        
            dataobj = np.zeros([parameters.antennasTX, parameters.antennasRX, data[d][0].shape[1]], dtype=complex)

            # Create complex values
            for tx in range(0,parameters.antennasTX):
                for rx in range(0,parameters.antennasRX):
                    for k in range(0,data[d][0].shape[1]):
                        dataobj[tx][rx][k] = data[d][0][2*(tx*parameters.antennasRX+rx)][k].item() + data[d][0][2*(tx*parameters.antennasRX+rx)+1][k].item() * 1j

            # Get the angles for all streams
            angle = AoA.applyMUSIC(dataobj, parameters.antennadist, parameters.wavelength, False, False)
            
            # Only consider the angles of three streams (instead of four)
            # The stream of the internal antenna is discarded!
            dataAoA.append([angle[0], angle[1], angle[3]])

    # Create a filename to store the AoA information in a separate file
    if parameters.getAoARefinement:
        filename = utils.createFileName(dataID=parameters.refineDataID+"_AoA", netID=-1, lossID=-1, extremes=-1, numExtremes=-1)
    else:
        filename = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # Store the calculated data
    utils.saveFile(filename, dataAoA)

    return [filename, dataLen, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Train a network with the specified architecture (determined by parameters.py)
def performTraining(errorLog=[]):

    # Load the training data
    data, dataLen, errorLog = essentialMethods.loadData(True, enumerators.DATA.TRAINING, True, errorLog)

    # initialize and train the network
    net, params, errorLog = learning.trainNetwork(data=data, numTraining=parameters.numTraining, batchsize=parameters.batchsize, droprate=parameters.droprate, kernelsize=parameters.kernelsize, 
        inchannels=parameters.inchannels, learningrate=parameters.learningrate, momentum=parameters.momentum, countepochs=parameters.epochsTotal, updatefreq=parameters.updatefreq, 
        printfreq=parameters.printfreq, lossfunction=parameters.lossfunctionTraining, normalizeData=parameters.normalizeData, backprop=parameters.backprop, 
        dropoutEnabled=parameters.dropoutEnabled, insertAoA=parameters.insertAoAInNetwork ,amountFeatures=parameters.amountFeatures,debug=parameters.debug, errorLog=errorLog)

    print("Training finished!")

    # Create filename with the given parameters (-1 = not present/not necessary)
    if(parameters.insertAoAInNetwork):
        filename = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
    else:
        filename = utils.createFileName(dataID=parameters.dataID, netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)

    # if debugging enabled, print filename
    if(parameters.debug):
        print(filename)

    # save net
    # Store corresponding configuration
    config = {"insertAoAInNetwork":parameters.insertAoAInNetwork,"selectFromSamplesPerPos":parameters.selectFromSamplesPerPos,
            "mask":parameters.mask,"shuffling":parameters.shuffling,"randomDigest":parameters.randomDigest, "params":params,
             "filterDataAfterLoading":parameters.filterDataAfterLoading, "excludedArea":parameters.excludedArea, "numTraining":parameters.numTraining,
             "normalizeData":parameters.normalizeData,"amountFeatures":parameters.amountFeatures, "antennasRXUsed":parameters.antennasRXUsed,
             "antennasTXUsed":parameters.antennasTXUsed, "samplesPerChannel":parameters.samplesPerChannel}


    utils.saveFile(filename,[net, config])

    # Disable dropout for testing (is only necessary if it was enabled previously)
    net.disableDropout()

    # calculate meanError in the TRAININGset
    meanError, errorLog = learning.testNetwork(data=data, net=net, numTraining=parameters.numTraining, batchsize=parameters.batchsize, lossfunction=parameters.lossfunctionTesting, useTrainingData=True, 
        normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
    print('Mean error in the TrainingSet with %d samples: %.3fm' % (parameters.numTraining,meanError*0.3))

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, dataLen, meanError, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Test a given network with certain data
def testNetwork(errorLog=[]):

    # Load first network, as it may set important configuration parameters
    net, params = essentialMethods.loadNet()

    # Print network architecture: uncomment to see it
    #utils.printInfo("Network Architecture: %s" % (str(net)))

    # Choose correct type of data
    if parameters.loadRefineDataAsTest:
        data, dataLen, errorLog = essentialMethods.loadRefinementData(True, errorLog)
    else:
        # Load network and data from files

        # If all data is required (not from the same day as the training)
        # Else: Use the data of the same day as the training, but only the set which was not used (= test set)
        if parameters.testAllData:
            data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True,errorLog)
        # Use the data of the same day as the training, but only the set which was not used (= test set)
        else:
            data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.TESTING, True,errorLog)

    # Calculate mean error in testdata
    meanError, errorLog = learning.testNetwork(data=data, net=net, numTraining=parameters.numTraining, batchsize=parameters.batchsize, lossfunction=parameters.lossfunctionTesting, useTrainingData=False, 
        normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
    print('Mean error in the TestSet with %d samples: %.3fm' % (len(data),meanError*0.3))
    
    return [-1, dataLen, meanError, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

def extremeSearch(errorLog=[]):

    # Load network and data from files

    # Extreme search either in the training or testset
    if(parameters.searchInTraining):
        datapart = enumerators.DATA.TRAINING
    else:
        datapart = enumerators.DATA.TESTING

    # Load the corresponding network
    net, params = essentialMethods.loadNet()

    if parameters.loadRefineDataAsTest:
        data, dataLen, errorLog = essentialMethods.loadRefinementData(True, errorLog)
    else:
        # Load network and data from files
        data, dataLen, errorLog = essentialMethods.loadData(False, datapart, True, errorLog)

    # Get the top xy score of best/worst loss
    extremes, estimatedIndices, errorLog = learning.getExtremeEstimates(data=data, net=net, numTraining=parameters.numTraining, lossfunction=parameters.lossfunctionTesting, numResult=parameters.numExtremes, bestResults=parameters.bestExtremes, 
                                            normalizeData=parameters.normalizeData, params=params, useTrainingData=parameters.searchInTraining, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, room=parameters.room, errorLog=errorLog)
    # Determine the set, which contains the indices of the best/worst highscore
            
    # Add to both sets the areas
    setRows = set()
    setCols = set()
    for i in range(0,len(extremes)):
        label = data[extremes[i][1]][1]
        
        index = parameters.room[int(label[0].item())][int(label[1].item())]
        
        for k in range(2,7):
            if index in parameters.rows[k]:
                setRows.add(enumerators.ROOM_ROW(k))
            if index in parameters.cols[k]:
                setCols.add(enumerators.ROOM_COL(k))
        
    foundcolareas = getSetContent(setCols ,"column")
    foundrowareas = getSetContent(setRows ,"row")

    # Print it on the terminal
    print("Top %d Ranking:" % (parameters.numExtremes))
    if(parameters.bestExtremes):
        for i in range(0,parameters.numExtremes):
            print("%2d: Best Error %.3fm (found at index %d)" % (i+1,extremes[i][0]*0.3, extremes[i][1]))  
    else:
        for i in range(0,parameters.numExtremes):
            print("%2d: Worst Error %.3fm (found at index %d)" % (i+1,extremes[i][0]*0.3, extremes[i][1]))     

    if(parameters.debug):
        print(filename)

    # Create filename with the given parameters (-1 = not present/not necessary)
    if(parameters.insertAoAInNetwork):
        filename = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=parameters.bestExtremes, numExtremes=parameters.numExtremes)
    else:
        filename = utils.createFileName(dataID=parameters.dataID, netID=parameters.netID, lossID=parameters.lossID, extremes=parameters.bestExtremes, numExtremes=parameters.numExtremes)

    utils.saveFile(filename, extremes)

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, dataLen, -1, [extremes, estimatedIndices], foundcolareas, foundrowareas, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

def analyzeTuneModel(errorLog=[]):

    # Load model and set the parameters correctly
    if parameters.loadRefinedNetwork:
        net, config = utils.loadFile(parameters.modelID+"_"+parameters.refineDataID)
    else:
        net, config = utils.loadFile(parameters.modelID)

    # Set all the parameters, which are stored in the config of the model
    parameters.insertAoAInNetwork = config["insertAoAInNetwork"]
    parameters.selectFromSamplesPerPos = config["selectFromSamplesPerPos"]
    parameters.mask = config["mask"]
    parameters.shuffling = config["shuffling"]
    parameters.randomDigest = config["randomDigest"]
    params = config["params"]
    parameters.filterDataAfterLoading = config["filterDataAfterLoading"]
    parameters.excludedArea = config["excludedArea"]
    parameters.numTraining = config["numTraining"]
    parameters.normalizeData = config["normalizeData"]
    parameters.amountFeatures = config["amountFeatures"]

    # These options are not contained in "old" configurations and therefore checked previously
    if "antennasRXUsed" in config:
        parameters.antennasRXUsed = config["antennasRXUsed"]
        parameters.antennasTXUsed = config["antennasTXUsed"]
        parameters.antennasRX = len(parameters.antennasRXUsed)
        parameters.antennasTX = len(parameters.antennasTXUsed)
        parameters.samplesPerChannel = config["samplesPerChannel"]

    # Print some information
    utils.printInfo("Analyzed model: %s" % (parameters.modelID))
    print(parameters.numTraining)

    # Uncomment this to see the architecture of the actual model
    #utils.printInfo("Network Architecture: %s" % (str(net)))

    parameters.netID = parameters.modelID
    
    # Use either all data (from a different day than training) to test it or only the testset
    if parameters.testAllData:
        data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True,errorLog)
    else:
    # Load data from files with set parameters TESTING
        data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.TESTING, True,errorLog)
     
    # This option is enabled -> also log the results and calculate the mean + quartiles -> needs more time
    if parameters.analyzeCDFAndLog:

        # Calculate mean error in testdata
        sortedOutputs, errorLog = learning.createSortedList(data=data, net=net, numTraining=parameters.numTraining, lossfunction=parameters.lossfunctionTesting, useTrainingData=False, 
            normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
        
        print("Mean is %.2f & Median is %.2f (%d samples)" % (np.mean(sortedOutputs), np.median(sortedOutputs), len(sortedOutputs)))

        results = plottingUtils.plotCumulative(sortedOutputs,"CDF of the Testingdata", True)
    else:
        # Calculate mean error in testdata: faster; not stored
        meanError, errorLog = learning.testNetwork(data=data, net=net, numTraining=parameters.numTraining, batchsize=parameters.batchsize, lossfunction=parameters.lossfunctionTesting, useTrainingData=False, 
        normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
        print('Mean error in the TestSet with %d samples: %.3fm' % (len(data),meanError*0.3))

        # Nothing in results, as not written in the log
        results = [-1, -1, -1]

    return [-1, dataLen, results, -1, -1, -1, errorLog]
  
# -------------------------------------------------------------------------------------------------------------------------------- #

# Provides a visualization of the data: where is the real location and where does the network puts the estimates?
def visualizeData(errorLog=[]):

    # Load network and data from files
    if(parameters.searchInTraining):
        datapart = enumerators.DATA.TRAINING
    elif(parameters.testAllData):
        datapart = enumerators.DATA.ALL
    else:
        datapart = enumerators.DATA.TESTING

    # Load the network, which should be used to estimate the positions
    net, params = essentialMethods.loadNet()

    if parameters.loadRefineDataAsTest:
        data, dataLen, errorLog = essentialMethods.loadRefinementData(True, errorLog)
    else:
        # Load network and data from files
        data, dataLen, errorLog = essentialMethods.loadData(False, datapart, True, errorLog)

    dataSet = dataManagement.CSIDataset(data, parameters.normalizeData, params)

    # Load extreme points, which were identified before
    # If this option is not chosen -> just use the first X data points
    if(not parameters.visualizeRandomData):

        # Create filename with the given parameters (-1 = not present/not necessary)
        if(parameters.insertAoAInNetwork):
            filename = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=parameters.bestExtremes, numExtremes=parameters.numExtremes)
        else:
            filename = utils.createFileName(dataID=parameters.dataID, netID=parameters.netID, lossID=parameters.lossID, extremes=parameters.bestExtremes, numExtremes=parameters.numExtremes)

        extremes = utils.loadFile(filename)

    outputs = []

    # add each point to outputs[]
    for i in range(0,parameters.numExtremes):
        netinput = torch.zeros(1,len(data[0][0]),len(data[0][0][0]))
        inputAoA = torch.zeros(1,3, dtype=float)

        # IMPORTANT: This can be modified manually: plot every X point -> e.g. index = 10 * i
        index = i
        if(not parameters.visualizeRandomData):
            index = extremes[i][1]

        netinput[0] = dataSet[index][0]
        label = dataSet[index][1]

        # If AoA is required -> add it!
        if(parameters.insertAoAInNetwork):
            inputAoA[0] = torch.from_numpy(dataSet[index][2])

        # get network output -> estimated
        out = net(netinput.float(), inputAoA.float())

        if(parameters.debug):
            print("Label: %s; Output of the network: %s" % (label, out))

        # round results and change the coordination format to make it more intuitive (change reference point)
        rounded_out = [round(out[0][0].item()), round(out[0][1].item())]
        out_coordChange = utils.changeCoordRef([rounded_out[0], rounded_out[1]], len(parameters.room))
        label_coordChange = utils.changeCoordRef([label[0].item(), label[1].item()], len(parameters.room))

        # calculate loss for actual point to display it on the terminal
        loss = learning.getLoss(parameters.lossfunctionTesting, out, label)

        print("Top %2d: The estimated positions was [%2d,%2d], while the true position is [%2d,%2d] (Error: %.3f)" 
            % (i+1, out_coordChange[0], out_coordChange[1], label_coordChange[0], label_coordChange[1], loss))

        # append it to the output struct
        outputs.append([rounded_out, [label[0].item(), label[1].item()]])

    processor = dataManagement.DataProcessing(parameters.debug)

    # outputs: first one = estimated, second = true position
    app = visualize.QApplication(sys.argv)

    # Display everything! all the outputs saved before to the outputs[] array
    x = visualize.roomVisualizer(parameters.roomName,processor.findPos(parameters.coordInfo[2], parameters.room),processor.findPos(parameters.coordInfo[3],parameters.room),outputs,parameters.room,-1, parameters.videoMode)
    sys.exit(app.exec_())

    # return all existing values (-1 = placeholder for not-existing)
    return [-1, -1, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Create a CDF plot to visualize the accuracy of the network on the passed data
def createCDF(errorLog=[]):

    # Load the network
    net, params = essentialMethods.loadNet()

    # If all data is required (not from the same day as the training)
    # Else: Use the data of the same day as the training, but only the set which was not used (= test set)
    if parameters.testAllData:
        data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True, errorLog)
    else:
        data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.TESTING, True, errorLog)

    # Calculate mean error in testdata
    sortedOutputs, errorLog = learning.createSortedList(data=data, net=net, numTraining=parameters.numTraining, lossfunction=parameters.lossfunctionTesting, useTrainingData=False, 
        normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
    
    # Print information
    print("Mean is %.2fm & Median is %.2fm (%d samples)" % (np.mean(sortedOutputs)*0.3, np.median(sortedOutputs)*0.3, len(sortedOutputs)))

    # Calls the plot (produces the image) and stores the quartiles
    # In the automated mode, the picture is not shown!
    results = plottingUtils.plotCumulative(sortedOutputs,"CDF of the Testingdata", parameters.automatedMode)

    # return all existing values (-1 = placeholder for not-existing)
    return [-1, dataLen, results, -1, -1, -1, errorLog]

# --------------------------------------------------------------------------------------------------------------------------------- #

# Create a heatmap to obtain a better understanding of the network performance on a given dataset
def createHeatMap(errorLog=[]):

    # Load the network / model 
    net, params = essentialMethods.loadNet()

    # Load the data
    data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True, errorLog)   #.TESTING

    # If desired, the data can be filtered (e.g. a certain amount of the grid is excluded)
    if(parameters.filterDataAfterLoading):
        excluded = parameters.excludedArea
    else:
        excluded=-1

    # In this case, data of multiple days / directories is used
    if isinstance(parameters.coordInfo[0], list):
        amountOfPos = parameters.coordInfo[0][1]
    else:
        amountOfPos = parameters.coordInfo[1]

    # Create a heatmap for the data
    heatmapObject = learning.createHeatMap(data=data, net=net, lossfunction=parameters.lossfunctionTesting, 
                                    amountOfPos=amountOfPos, params=params, insertAoA=parameters.insertAoAInNetwork, 
                                    debug=parameters.debug, room= parameters.room, excluded=excluded, normalizeData=parameters.normalizeData)
    # Some debug information
    if(parameters.debug):
        print(heatmapObject)

    # Divide the heatmap object into the map itself and information regarding the border / min and max
    heatmap, borders = heatmapObject

    # init dummy datamanagement object to get access to simple data functions
    dataProcessor = dataManagement.DataProcessing(False)

    # Init QApp
    app = visualize.QApplication(sys.argv)

    # Create color array from red to green
    green = Color("green")
    yellow = Color("yellow")
    colors = list(green.range_to(Color("yellow"),250)) + list(yellow.range_to(Color("red"), 250))

    # In this case, data of multiple days / directories is used
    if isinstance(parameters.coordInfo[0], list):    
        # Displays the data
        x = visualize.heatMapVisualizer(parameters.roomName, dataProcessor.findPos(parameters.coordInfo[0][2], parameters.room),
            dataProcessor.findPos(parameters.coordInfo[0][3],parameters.room),heatmap, parameters.room, colors, borders, parameters.debug, True, parameters.heatMapTimeout, "HeatMap_%s" % (parameters.netID))
    else:
        # Displays the data
        x = visualize.heatMapVisualizer(parameters.roomName, dataProcessor.findPos(parameters.coordInfo[2], parameters.room),
            dataProcessor.findPos(parameters.coordInfo[3],parameters.room),heatmap, parameters.room, colors, borders, parameters.debug, True, parameters.heatMapTimeout, "HeatMap_%s" % (parameters.netID))

    # In the automated mode, the app is closed automatically
    if(parameters.automatedMode):
        app.exec_()
        app.quit()
    else:
        # Closed by the user
        sys.exit(app.exec_())

    # return all existing values (-1 = placeholder for not-existing)
    return [-1, dataLen, -1, -1, -1, -1, errorLog]

# --------------------------------------------------------------------------------------------------------------------------------- #

# Create a hitmap which shows where a network puts the estimates
def createHitMap(errorLog=[]):

    # Load the network / model 
    net, params = essentialMethods.loadNet()

    # Load network and data from files
    data, dataLen, errorLog = essentialMethods.loadData(False, enumerators.DATA.ALL, True, errorLog)   #.TESTING

    # If desired, the data can be filtered (e.g. a certain amount of the grid is excluded)
    if(parameters.filterDataAfterLoading):
        excluded = parameters.excludedArea
    else:
        excluded=-1

    # In this case, data of multiple days / directories is used
    if isinstance(parameters.coordInfo[0], list):
        amountOfPos = parameters.coordInfo[0][1]
    else:
        amountOfPos = parameters.coordInfo[1]

    # Create either hitmaps for each single position where measurements were taken
    # or: if allDataHits is chosen: Visualize all estimates independent of real position
    if parameters.visualizeAllDataHits:
        hitMap = learning.createHitMap(data=data, net=net, lossfunction=parameters.lossfunctionTesting, 
                                    amountOfPos=amountOfPos, params=params, insertAoA=parameters.insertAoAInNetwork, 
                                    debug=parameters.debug, room= parameters.room, excluded=excluded, normalizeData=parameters.normalizeData, forAll=True)
    else:
        hitMap = learning.createHitMap(data=data, net=net, lossfunction=parameters.lossfunctionTesting, 
                                    amountOfPos=amountOfPos, params=params, insertAoA=parameters.insertAoAInNetwork, 
                                    debug=parameters.debug, room= parameters.room, excluded=excluded, normalizeData=parameters.normalizeData, forAll=False)
    
    # Debug: Show the returned object
    if(parameters.debug):
        print(heatmapObject)

    # init dummy datamanagement object to get access to simple data functions
    dataProcessor = dataManagement.DataProcessing(False)

    # Init QApp
    app = visualize.QApplication(sys.argv)

    # Create color array from red to green
    white = Color("white")
    yellow = Color("black")
    colors = list(white.range_to(Color("black"),500))# + list(yellow.range_to(Color("red"), 300))

    # In this case, data of multiple days / directories is used
    if isinstance(parameters.coordInfo[0], list):    
        # Displays the data
        x = visualize.hitMapVisualizer(parameters.roomName, dataProcessor.findPos(parameters.coordInfo[0][2], parameters.room),
            dataProcessor.findPos(parameters.coordInfo[0][3],parameters.room),hitMap, parameters.room, colors, parameters.debug, parameters.heatMapTimeout, "HeatMap_%s" % (parameters.netID))
    else:
        # Displays the data
        x = visualize.hitMapVisualizer(parameters.roomName, dataProcessor.findPos(parameters.coordInfo[2], parameters.room),
            dataProcessor.findPos(parameters.coordInfo[3],parameters.room),hitMap, parameters.room, colors, parameters.debug, parameters.heatMapTimeout, "HeatMap_%s" % (parameters.netID))

    # In the automated mode, the app is closed automatically
    if(parameters.automatedMode):
        app.exec_()
        app.quit()
    else:
        sys.exit(app.exec_())

    # return all existing values (-1 = placeholder for not-existing)
    return [-1, dataLen, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Load Refinement Data:
# Combines Data loading and applies the baseline as well
# This data is later used to retrain/ refine a network
def loadRefinementData(errorLog=[]):

    # Get the data as well as params (length of one (or multiple directionaries))
    data, dataLen, errorLog = dataManagement.getDataFromRawPipeline(directory=parameters.directoryRefinement, coordInfo=parameters.coordInfoRefinement, room=parameters.roomRefinement, coordMode=parameters.coordMode,
                      filesPerTarget=parameters.filesPerTargetRefinement, signalrepr=parameters.signalrepr, exception=set(), 
                      pilots=parameters.pilots, datarange=parameters.datarange, debug=parameters.debug, rmPilots=parameters.rmPilots, rmUnused=parameters.rmUnused,
                      samplesToAnalyze=parameters.samplesToAnalyze, converter=parameters.importMethod, inchannels=parameters.inchannels, errorLog=errorLog)

    if(parameters.debug):
        print(filename)

    # If desired, normalize the data against the baseline
    if parameters.useCombinedData :

        # Apply Coax/Base if necessary
        base, baseLen = essentialMethods.loadBase()
        reference = base[parameters.baseNo][0]

        # Convert into complex (IQ) samples and then apply baseline
        for i in range(0,dataLen):
            for k in range(0,len(reference)//2):
                ref = np.array([np.add(reference[2*k].numpy(), np.multiply(reference[2*k+1].numpy(), 1j))], dtype=complex)
                dataSig = np.array([np.add(data[i][0][2*k].numpy(), np.multiply(data[i][0][2*k+1].numpy(), 1j))], dtype=complex)
                normalizedData = dataSig / ref
                data[i][0][2*k] = torch.from_numpy(np.real(normalizedData))
                data[i][0][2*k+1] = torch.from_numpy(np.imag(normalizedData))

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.refineDataID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # Save data
    utils.saveFile(filename,data)

    # return all existing values (-1 = placeholder for not-existing)
    return [filename, dataLen, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# Refine a certain network with the data, which was loaded previously
# While the convolutional layers are frozen, the fully connected ones are retrained/ weights adapted
def performRefinement(errorLog=[]):

    # Load first network, as it may set important configuration parameters
    basic_net, basic_params = essentialMethods.loadNet()

    # This will be appended to store the refined network
    saveAs = parameters.refineDataID

    # in automatedMode, multiple trainings are done and the savedID should be used
    if(parameters.automatedMode):
        parameters.refineDataID = parameters.refineDataIDSaved

    # Load the refinement data
    data, dataLen, errorLog = essentialMethods.loadRefinementData(True, errorLog)

    # Shuffle all the data
    random.shuffle(data)

    # Split into training and test
    trainingsdata = data[:parameters.refinementNumTraining]
    testdata = data[parameters.refinementNumTraining:]

    # Refine the network
    net, params, errorLog = learning.refineNetwork(data=trainingsdata, net=basic_net, params=basic_params, batchsize=parameters.batchsize,
        countepochs=parameters.epochsTotal, printfreq=parameters.printfreq, lossfunction=parameters.lossfunctionTraining, loadModelAsNet=parameters.loadModelAsNet, learningrate=parameters.learningrate,
        normalizeData=parameters.normalizeData, backprop=parameters.backprop, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)

    print("Refinement finished!")

    # If a model is used -> use loadModelAsNet instead of NetID
    if(len(parameters.loadModelAsNet) > 1):
        filename = parameters.loadModelAsNet+"_"+saveAs

        # Store corresponding configuration
        config = {"insertAoAInNetwork":parameters.insertAoAInNetwork,"selectFromSamplesPerPos":parameters.selectFromSamplesPerPos,
                "mask":parameters.mask,"shuffling":parameters.shuffling,"randomDigest":parameters.randomDigest, "params":params,
                 "filterDataAfterLoading":parameters.filterDataAfterLoading, "excludedArea":parameters.excludedArea, "numTraining":parameters.numTraining,
                 "normalizeData":parameters.normalizeData,"amountFeatures":parameters.amountFeatures, "antennasRXUsed":parameters.antennasRXUsed,
                 "antennasTXUsed":parameters.antennasTXUsed, "samplesPerChannel":parameters.samplesPerChannel}

        utils.saveFile(filename,[net,config])

    else:
        # Create filename with the given parameters (-1 = not present/not necessary)
        if(parameters.insertAoAInNetwork):
            filename = utils.createFileName(dataID=parameters.dataID+"_"+saveAs+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
        else:
            filename = utils.createFileName(dataID=parameters.dataID+"_"+saveAs, netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)

        # Store corresponding configuration
        config = {"insertAoAInNetwork":parameters.insertAoAInNetwork,"selectFromSamplesPerPos":parameters.selectFromSamplesPerPos,
                "mask":parameters.mask,"shuffling":parameters.shuffling,"randomDigest":parameters.randomDigest, "params":params,
                 "filterDataAfterLoading":parameters.filterDataAfterLoading, "excludedArea":parameters.excludedArea, "numTraining":parameters.numTraining,
                 "normalizeData":parameters.normalizeData,"amountFeatures":parameters.amountFeatures, "antennasRXUsed":parameters.antennasRXUsed,
                 "antennasTXUsed":parameters.antennasTXUsed, "samplesPerChannel":parameters.samplesPerChannel}

        # save net
        utils.saveFile(filename,[net, config])

    if(parameters.debug):
        print(filename)

    # Disable dropout (independent of the fact if it was used or enabled prev.)
    net.disableDropout()

    # calculate meanError in the Testset
    meanError, errorLog = learning.testNetwork(data=testdata, net=net, numTraining=len(data), batchsize=parameters.batchsize, lossfunction=parameters.lossfunctionTesting, useTrainingData=False, 
        normalizeData=parameters.normalizeData, params=params, insertAoA=parameters.insertAoAInNetwork, debug=parameters.debug, errorLog=errorLog)
    print('Mean error in the Testset with %d samples: %.3fm' % (dataLen - parameters.refinementNumTraining,meanError*0.3))
    
    # return all existing values (-1 = placeholder for not-existing)
    return [filename, dataLen, meanError, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# The instant mode uses a trained network to predict position in real time
def instantMode(errorLog=[]):

    # Load the network
    net, params = essentialMethods.loadNet()

    # Get the directory
    directory = "Instant/"
    datadirectory = dataParameters.directory
    os.chdir(datadirectory+"/")

    # Get the actual amount of files in the given directory (Instant)
    before = dict([(f, None) for f in os.listdir(directory)])

    # Check if baseline necessary (coaxMeasurement or emptyRoom)
    if(parameters.useCombinedData):
        base, baseLen = essentialMethods.loadBase()
        reference = base[parameters.baseNo][0]  # Choose the correct reference sample

    # Do not try to open old files
    state = -1

    while 1:
        # Change the directory in the data directory
        os.chdir(datadirectory+"/")

        # Check the directory for new files (live from the router every second)
        time.sleep(1)

        # Actual files
        after = dict([(f, None) for f in os.listdir(directory)])

        # Check for new files (added)
        added = [f for f in after if not f in before]

        # Treat all new measurements from the router
        for filename in added:
            if ".hdf5" in filename: # Only accept the correct format

                print(filename.split('_')[1])

                # Check if it is an old file (already handled)
                if(state > int(filename.split('_')[1])):
                    print("Old file detected (State is %d)" % (state))
                    continue
                else:
                    state = int(filename.split('_')[1]) + 1

                time.sleep(1)
                print("Analyzed file: %s" % (filename))

                # Load the file 
                data, datalen = dataManagement.getInstantData(directory=directory, filename=filename, signalrepr=parameters.signalrepr, 
                                                            exception=parameters.excludedArea, pilots=parameters.pilots, 
                                                            datarange=parameters.datarange, debug=parameters.debug, rmPilots=parameters.rmPilots, rmUnused=parameters.rmUnused, 
                                                            room=parameters.room, converter=parameters.importMethod, packetsToCombine=parameters.packetsToCombine)

                # If an error occured (e.g. not as many packets as desired -> delete all these packets)
                if(data == -1):
                    utils.printInfo("Router did not send the correct amount of packets")

                    # Delete all the files
                    for i in range(1,parameters.packetsToCombine+1+1):

                        filename = filename.split("__")[0] + "__" + str(i) + ".hdf5"
                        try:
                            os.remove(os.path.join(directory,filename))
                            print(filename+" removed")
                        except OSError:
                            continue

                    # Skip the rest and continue with the next file
                    continue
                
                # Average the data if necessary
                if(parameters.packetsToCombine > 1 and parameters.averageInput):
                    tensorToAverage = torch.Tensor(parameters.packetsToCombine, len(data[0][0]), len(data[0][0][0]))
                    for index in range (0,parameters.packetsToCombine):
                        tensorToAverage[index] = data[index][0]

                    averaged = torch.mean(tensorToAverage,0)

                    data = [[averaged,data[0][1]]]
                

                # Check if baseline / coaxMeasurement necessary
                if(parameters.useCombinedData):

                    for dataindex in range(0,len(data)):
                        for k in range(0,len(reference)//2):
                            ref = np.array([np.add(reference[2*k].numpy(), np.multiply(reference[2*k+1].numpy(), 1j))], dtype=complex)
                            dataSig = np.array([np.add(data[dataindex][0][2*k].numpy(), np.multiply(data[dataindex][0][2*k+1].numpy(), 1j))], dtype=complex)
                            normalizedData = dataSig / ref
                    
                            data[dataindex][0][2*k] = torch.from_numpy(np.real(normalizedData))
                            data[dataindex][0][2*k+1] = torch.from_numpy(np.imag(normalizedData))
                    os.chdir(datadirectory+"/")
                
                # Normalize if necessary
                dataSet = dataManagement.CSIDataset(data, parameters.normalizeData, params)
                print("DataLen: %d & Len dataSet: %d" % (datalen, len(dataSet)))

                if(datalen != parameters.packetsToCombine):
                    utils.printFailure("Wrong amounts of packets!")

                outputs = np.zeros([len(data),2])

                # For all the measurements estimate the positions
                for dataindex in range(0,len(data)):

                    netinput = torch.zeros(1,len(data[0][0]),len(data[0][0][0]))
                    inputAoA = torch.zeros(1,3)

                    netinput[0] = dataSet[dataindex][0]

                    # Calculate AoA if necessary
                    if(parameters.insertAoAInNetwork):
                        dataobj = np.zeros([parameters.antennasTX, parameters.antennasRX, data[0][0].shape[1]], dtype=complex)

                        for tx in range(0,parameters.antennasTX):
                            for rx in range(0,parameters.antennasRX):
                                for k in range(0,data[0][0].shape[1]):
                                    dataobj[tx][rx][k] = data[dataindex][0][2*(tx*parameters.antennasRX+rx)][k].item() + data[dataindex][0][2*(tx*parameters.antennasRX+rx)+1][k].item() * 1j

                
                        angle = AoA.applyMUSIC(dataobj, parameters.antennadist, parameters.wavelength, False, False)
                        result = np.real(angle);
                        print(result)

                        inputAoA[0] = torch.from_numpy(np.array([result[0], result[1], result[3]]))

                    # get network output -> estimated
                    out = net(netinput, inputAoA)

                    # Adapt coordinates to visual output: 
                    rounded_out = [round(out[0][0].item()), round(out[0][1].item())]
                    outputs[dataindex,:] = rounded_out
                    output = "Estimated position is %d - %d" % (rounded_out[1]+1, rounded_out[0]+1)

                    print(output)

                # Average if desired
                if(len(data) > 1):
                    averagedoutput = np.average(outputs,0)

                    output = "Estimated position is %d - %d" % (averagedoutput[1]+1, averagedoutput[0]+1)

                # Now read it out loud -> text is read faster and the actual number slower
                engine = pyttsx3.init()
                engine.setProperty('rate',150)
                engine.say(output[:21])

                engine.setProperty('rate',30)
                engine.say(output[21:])

                engine.runAndWait()

                # Some debug parameters
                if(parameters.debug):
                    rate = engine.getProperty('rate')
                    volume = engine.getProperty('volume')
                    voice = engine.getProperty('voice')
                    print("Rate: ", rate, " Volume: ", volume, " Voice: ", voice)
                
                # Then display it in the visualization room for 1 second (close it afterwards)
                outputs=[]        

                # append it to the output struct
                outputs.append([rounded_out, [-1, -1]])

                processor = dataManagement.DataProcessing(parameters.debug)

                # outputs: first one = estimated, second = true position
                app = visualize.QApplication(sys.argv)

                # Display everything! all the outputs saved before to the outputs[] array
                x = visualize.roomVisualizer(parameters.roomName,processor.findPos(parameters.coordInfo[2], parameters.room, parameters.videoMode),
                    processor.findPos(parameters.coordInfo[3],parameters.room),outputs,parameters.room,parameters.timeout_Instant)
                app.exec_()
                app.quit()

                # delete unused objects
                del app
                del x

                # In the end: Remove the handled files
                for i in range(1,parameters.packetsToCombine+1+1):

                    filename = filename.split("__")[0] + "__" + str(i) + ".hdf5"

                    # Can only be removed if existing
                    try:
                        os.remove(os.path.join(directory,filename))
                        print(filename+" removed")
                    except OSError:
                        continue
                   
        # Update current amount of files in the directory
        before=after

    return [-1, -1, -1, -1, -1, -1, errorLog]

# -------------------------------------------------------------------------------------------------------------------------------- #

# This method is for experiments etc. 
# Will not be logged
def userDefined(errorLog=[]):
    
    print("User Defined Mode called.")

    # return all existing values (-1 = placeholder for not-existing)
    return [-1, -1, -1, -1, -1, -1, errorLog]

#####################################################################################################################################

def main():

    print("Localization Framework is starting... (Cycles: %d)" % (len(parameters.modes)))

    timeStart = time.perf_counter()

    # Error Log to collect all the error messages and print them again in the end
    errorLog = []

    for i in range(0,len(parameters.modes)):

        print("#############################")
        print("CYCLE NUMBER %d (of %d)" % (i+1, len(parameters.modes)))
        print("#############################")

        utils.printWithTime("Current Time after %d iterations:" % (i), timeStart)

        # Update the Mode
        mode = parameters.modes[i]

        # If automated -> Update Parameters
        if(parameters.automatedMode):
            updateParameters(i)

        # Measure the time
        timeNow = str(datetime.now())
        t0 = time.perf_counter()

# ----------------------------------------------- CHOOSE MODE ACCORDING TO SELECTED ONE ------------------------------------------ #
        
        # If the mode was saved somewhere, the methods return the savedname, if not: return -1
        if(mode == enumerators.MODE.DATALOADER):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = dataProcessing(errorLog)
        elif(mode == enumerators.MODE.BASELOADER):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = baselineProcessing(errorLog)
        elif(mode == enumerators.MODE.COAXLOADER):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = loadCoax(errorLog)
        elif(mode == enumerators.MODE.APPLYMASK):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = applyMask(errorLog)            
        elif(mode == enumerators.MODE.APPLYBASETODATA):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = applyBaseLineToData(errorLog)
        elif(mode == enumerators.MODE.GETAOAOFDATA):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = getAoAOfData(errorLog)
        elif(mode == enumerators.MODE.TRAINING):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = performTraining(errorLog)
        elif(mode == enumerators.MODE.TESTING):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = testNetwork(errorLog)
        elif(mode == enumerators.MODE.EXTREMESEARCH):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = extremeSearch(errorLog)
        elif(mode == enumerators.MODE.ANALYZETUNEMODELS):
            parameters.modelID = parameters.modelIDs[i]
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = analyzeTuneModel(errorLog)
        elif(mode == enumerators.MODE.VISUALIZATION):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = visualizeData(errorLog)
        elif(mode == enumerators.MODE.CDFPLOT):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = createCDF(errorLog)
        elif(mode == enumerators.MODE.CREATEHEATMAP):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = createHeatMap(errorLog)
        elif(mode == enumerators.MODE.CREATEHITMAP):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = createHitMap(errorLog)
        elif(mode == enumerators.MODE.REFINEDATALOADER):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = loadRefinementData(errorLog)
        elif(mode == enumerators.MODE.REFINENETWORK):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = performRefinement(errorLog)
        elif(mode == enumerators.MODE.INSTANT):
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = instantMode(errorLog)
        else:
            savedname, dataLen, meanError, extremes, foundcolareas, foundrowareas, errorLog = userDefined(errorLog)
        

# ------------------------------------------------- MEASURE TIME ---------------------------------------------------------------- #

        # Log the elapsed time
        t1 = time.perf_counter() - t0

# ------------------------------------------------- CREATE LOG IF DESIRED ------------------------------------------------------- #

        if(parameters.loggingEnabled):
            utils.writeLog(mode=mode, savedname=savedname, logfilename=parameters.logFile, directory=parameters.directory, dataID=parameters.dataID, netID=parameters.netID, lossID=parameters.lossID,
                time=timeNow, elapsedTime=t1, roomName=parameters.roomName, excluded=parameters.excluded, officeRows=parameters.exceptRow, officeCols=parameters.exceptCol, 
                coordInfo=parameters.coordInfo, coordMode=parameters.coordMode, signalrepr=parameters.signalrepr, 
                samplesPerTarget=parameters.filesPerTarget, learningrate=parameters.learningrate, momentum=parameters.momentum, batchsize=parameters.batchsize, updatefreq=parameters.updatefreq, 
                droprate=parameters.droprate, kernelsize=parameters.kernelsize, inchannels=parameters.inchannels, epochsTotal=parameters.epochsTotal, 
                lossfunctionTraining=parameters.lossfunctionTraining, lossfunctionTesting=parameters.lossfunctionTesting, numTraining=parameters.numTraining, numTotal=dataLen, insertedAoa=parameters.insertAoAInNetwork, meanError=meanError, extremes=extremes, 
                bestExtremes=parameters.bestExtremes, foundcolareas=foundcolareas, foundrowareas=foundrowareas, backprop=parameters.backprop, width=parameters.logWidth,
                baseID=parameters.baseID, base_files=parameters.base_files, base_directory=parameters.base_directory, base_config=parameters.base_config,
                coaxInfo=parameters.coax_config, coax_directory=parameters.coax_directory, coaxFilesPerAntenna=parameters.coaxFilesPerAntenna,
                dropoutEnabled=parameters.dropoutEnabled, searchInTraining=parameters.searchInTraining, selectFromSamplesPerPos=parameters.selectFromSamplesPerPos,
                normalize=parameters.normalizeData, shuffling=parameters.shuffling, rmPilots=parameters.rmPilots, 
                rmUnused=parameters.rmUnused, usePrevRandomVal=parameters.usePrevRandomVal, baseNo=parameters.baseNo, filterDataAfterLoading=parameters.filterDataAfterLoading, maskID=parameters.maskID,
                randomVal=parameters.randomDigest, useCombinedData=parameters.useCombinedData,amountFeatures=parameters.amountFeatures,analyzeCDFAndLog=parameters.analyzeCDFAndLog)

# ---------------------------------------------------- RETURN FOUND ERRORS ------------------------------------------------------- #

    # Log all errors in the end together, if the errorLog was used
    # This makes it easier to debug it
    print('%sFound errors: %s' % (fg(11), attr(0)))

    for i in range (0,len(errorLog)):
        utils.printFailure('- %d. Error: %s' % (i+1,errorLog[i]))

    if(len(errorLog) < 1):
        utils.printSuccess("No Errors found :-)")

# ----------------------------------------------------------- MAIN METHOD -------------------------------------------------------- #

if __name__ == "__main__":
    main()

####################################################################################################################################
