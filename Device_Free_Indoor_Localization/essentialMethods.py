#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides methods to load the network, baseline, data or data to refine the network

This file is called essentialMethods, as it contains all necessary methods from the main.py file
Therefore, only this file is needed for the optimizationTune file

"""

############################################## IMPORTS #############################################################

import os
import math
import time
import pickle
import sys
import hashlib
import random
import torch
import sys

from colored import fg, bg, attr    # More information: https://pypi.org/project/colored/
import numpy as np

import enumerators
import utils
import learning
import dataManagement
import parameters

########################################### LOAD REFINEMENT DATA ##################################################

# Load data for refinement of an existing network
def loadRefinementData(considerAoA,errorLog=[]):

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.refineDataID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # load data object
    data = utils.loadFile(filename)
    dataLen= len(data)

    # If AoA should be used and loaded 
    if(considerAoA and parameters.insertAoAInNetwork):

        # Load AoA File
        filenameAoA = utils.createFileName(dataID=parameters.refineDataID+"_AoA", netID=-1, lossID=-1, extremes=-1, numExtremes=-1)
        data_AoA = utils.loadFile(filenameAoA)

        # Add AoA Information to the data
        for i in range(0,len(data)):
            data[i] = [data[i][0], data[i][1], np.real(data_AoA[i])]


    return data, dataLen, errorLog


################################################# LOAD DATA #######################################################

# Load data object and shrink channels if necessary
def loadData(forTraining, DATAPART, considerAoA, errorLog=[]):

    # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.dataID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # load data object
    data = utils.loadFile(filename)
    dataLen= len(data)

    data_rest = []

    print("%s loaded (%d samples)" % (filename, dataLen))

    # If AoA should be used and loaded 
    if(considerAoA and parameters.insertAoAInNetwork):

        # Load AoA File
        filenameAoA = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=-1, lossID=-1, extremes=-1, numExtremes=-1)
        data_AoA = utils.loadFile(filenameAoA)

        # Add AoA Information to the data
        for i in range(0,len(data)):
            data[i] = [data[i][0], data[i][1], np.real(data_AoA[i])]

    # make the space smaller, from which the training samples should be selected
    
    if(DATAPART != enumerators.DATA.ALL and parameters.selectFromSamplesPerPos < parameters.filesPerTarget):
        data_filtered = []

        print("Data filtered Samples per Position")

        # Iterate over all and split the data into data (used for training) and data_rest (used for testing later)
        for i in range(0,dataLen // parameters.filesPerTarget):
            for j in range(0,parameters.selectFromSamplesPerPos):
                data_filtered.append(data[i*parameters.filesPerTarget + j])
            for j2 in range(parameters.selectFromSamplesPerPos, parameters.filesPerTarget):
                data_rest.append(data[i*parameters.filesPerTarget + j2])

        # assign it to data
        data = data_filtered
        dataLen = len(data_filtered)
    
    # Apply mask -> should be tested
    if(not isinstance(parameters.mask, int)):
        data, errorLog = dataManagement.applyMaskOnData(data=data, coordInfo=parameters.coordInfo, filesPerTarget=parameters.filesPerTarget, room=parameters.room, mask=parameters.mask, debug=parameters.debug, errorLog=errorLog)
    
    # Store shuffled data
    shuffled_data = []

    # If a training is done and not the prev value is used -> new random permutation necessary
    if(forTraining and parameters.shuffling and not parameters.usePrevRandomVal):

        # Get random permutation
        random_indices = np.random.permutation(dataLen)
        
        # Add the data permutated
        for i in range(0,dataLen):
            shuffled_data.append(data[random_indices[i]])
        
        data = shuffled_data

        # Create Hash to describe the random permutation and store it
        sha256 = hashlib.sha256()
        sha256.update(bytes(random_indices))
        digestName = sha256.hexdigest()[:10]
        utils.saveFile("R_"+digestName,random_indices)
        parameters.randomDigest = digestName

        # As it is for a training, only make use of the first entires (length = amount of training samples)
        data = shuffled_data[:parameters.numTraining]
        dataLen = len(data)

    # If the whole data should be used -> no random permutation necessary
    elif(DATAPART == enumerators.DATA.ALL):

        # no shuffling necessary as all the data requested
        dataLen = len(data)

    # Either the testing or training is requested and shuffling should be applied (based on prev permutation)
    elif(parameters.shuffling):
    
        print(len(data))

        # Load permutation information
        random_indices = utils.loadFile("R_"+parameters.randomDigest)

        # Permutate the data according to permutation
        for i in range(0,len(random_indices)):
            shuffled_data.append(data[random_indices[i]])
        
        # Filter according to requested set
        if(DATAPART == enumerators.DATA.TRAINING):
            data = shuffled_data[:parameters.numTraining]
            dataLen = len(data)
        elif(DATAPART == enumerators.DATA.TESTING):
            # Also add the data, which was ignored previously for trainingsample selection!
            data_rest.extend(shuffled_data[parameters.numTraining:])
            data = data_rest
            dataLen = len(data)
    
    # No shuffling
    else:

        # Filter according to requested set
        if(DATAPART == enumerators.DATA.TRAINING):
            data = data[:parameters.numTraining]
            dataLen = len(data)
        elif(DATAPART == enumerators.DATA.TESTING):#
            # Also add the data, which was ignored previously for trainingsample selection!
            data_rest.extend(data[parameters.numTraining:])
            data = data_rest
            dataLen = len(data)   

    #First filter after TX, then RX 
    if(parameters.antennasTX < 4 or parameters.antennasRX < 4):
        for item in range(0,len(data)):
            # Init with fake data (deleted in the end)
            dataobj = torch.zeros([1,len(data[0][0][0])])
            
            if(parameters.antennasTX < 4):
                for i in range(0,4):
                    if i+1 in parameters.antennasTXUsed:
                        dataobj = torch.cat((dataobj.to(torch.float32), data[item][0][i*4*2:(i+1)*4*2].to(torch.float32)), dim=0).to(torch.float32)

            # Delete zero row
            dataobj = dataobj[1:]
            
            if(parameters.antennasRX < 4):
                dataobj_new = torch.zeros([1,len(data[0][0][0])])
                for tx in range(0,parameters.antennasTX):
                    for i in range(0,4):
                        if i+1 in parameters.antennasRXUsed:
                            dataobj_new = torch.cat((dataobj_new.to(torch.float32), dataobj[tx*4*2+2*i:tx*4*2+2*(i+1)]),dim=0).to(torch.float32)

                # delete zero row
                dataobj = dataobj_new[1:]

            data[item][0] = dataobj     

    # Apply filtering of Subcarrier! Use less to compare against other paper!
    if(len(data[0][0][0]) > parameters.samplesPerChannel):
        for i in range(0,len(data)):
            data[i][0] = data[i][0][:,0:parameters.samplesPerChannel]
   
    # Filter the data before loading into the network
    if(parameters.filterDataAfterLoading):
        data = dataManagement.filterData(data=data, room=parameters.room, excludedArea=parameters.excludedArea)
        dataLen=len(data)

    return data, dataLen, errorLog

################################################# LOAD NET ########################################################

# Load Net object 
def loadNet():

    if(len(parameters.loadModelAsNet)) > 1:

        # Load model and set the parameters correctly
        if parameters.loadRefinedNetwork:
            net, config = utils.loadFile(parameters.loadModelAsNet+"_"+parameters.refineDataID)
        else:
            print(parameters.loadModelAsNet)
            net, config = utils.loadFile(parameters.loadModelAsNet)

        if "antennasRXUsed" in config:
            parameters.antennasRXUsed = config["antennasRXUsed"]
            parameters.antennasTXUsed = config["antennasTXUsed"]
            parameters.antennasRX = len(parameters.antennasRXUsed)
            parameters.antennasTX = len(parameters.antennasTXUsed)
            parameters.samplesPerChannel = config["samplesPerChannel"]

        parameters.insertAoAInNetwork = config["insertAoAInNetwork"]
        parameters.selectFromSamplesPerPos = config["selectFromSamplesPerPos"]
        parameters.mask = config["mask"]
        parameters.shuffling = config["shuffling"]
        parameters.randomDigest = config["randomDigest"]
        params = config["params"]
        parameters.numTraining = config["numTraining"]
        parameters.normalizeData = config["normalizeData"]
        parameters.amountFeatures = config["amountFeatures"]

        if parameters.loadRefinedNetwork:
            print(parameters.loadModelAsNet+"_"+parameters.refineDataID+" loaded!")
        else:
            print(parameters.loadModelAsNet+" loaded!")

    else:
        # Create filename with the given parameters (-1 = not present/not necessary)
        if(parameters.insertAoAInNetwork):
            if parameters.loadRefinedNetwork:
                filename = utils.createFileName(dataID=parameters.dataID+"_"+parameters.refineDataID+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
            else:
                filename = utils.createFileName(dataID=parameters.dataID+"_AoA", netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
        else:
            if parameters.loadRefinedNetwork:
                filename = utils.createFileName(dataID=parameters.dataID+"_"+parameters.refineDataID, netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
            else:
                filename = utils.createFileName(dataID=parameters.dataID, netID=parameters.netID, lossID=parameters.lossID, extremes=-1, numExtremes=-1)
        
        # load netobject and parameter
        net, config = utils.loadFile(filename)

        parameters.antennasRXUsed = config["antennasRXUsed"]
        parameters.antennasTXUsed = config["antennasTXUsed"]
        parameters.antennasRX = len(parameters.antennasRXUsed)
        parameters.antennasTX = len(parameters.antennasTXUsed)
        parameters.samplesPerChannel = config["samplesPerChannel"]

        parameters.insertAoAInNetwork = config["insertAoAInNetwork"]
        parameters.selectFromSamplesPerPos = config["selectFromSamplesPerPos"]
        parameters.mask = config["mask"]
        parameters.shuffling = config["shuffling"]
        parameters.randomDigest = config["randomDigest"]
        params = config["params"]
        parameters.numTraining = config["numTraining"]
        parameters.normalizeData = config["normalizeData"]
        parameters.amountFeatures = config["amountFeatures"]

    # disable dropout to make the network deterministic
    net.disableDropout()

    # return all existing values (-1 = placeholder for not-existing)
    return [net, params]

################################################# LOAD BASE #######################################################

# Load baseline
def loadBase():

     # Create filename with the given parameters (-1 = not present/not necessary)
    filename = utils.createFileName(dataID=parameters.baseID, netID=-1, lossID=-1, extremes=-1, numExtremes=-1)

    # load data object
    data = utils.loadFile(filename)
    dataLen= len(data)

    return data, dataLen

###################################################################################################################