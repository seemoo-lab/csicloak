#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

"""

# ----------------- Imports ----------------------- #

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

import dataManagement
import enumerators
import plottingUtils
import utils

# ------------------------------------------------- #

# Source for the basic network structure:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

#############################################################################################################
################################################ NET ########################################################
#############################################################################################################

# Class Net: defines a neural network
class Net(nn.Module):
    
# ---------------------------------------- DEFINE NETWORK ---------------------------------------------------------- #

    # initialize network
    # droprate only important, if dropoutEnabled
    def __init__(self, droprate, kernelsize, inchannels, samplesPerChannel, dropoutEnabled, debug, AoAenabled, amountFeatures):
        super(Net, self).__init__()

        # AoA enabled -> insert 3 additional units in AoA
        self.AoAenabled = AoAenabled

        if(AoAenabled):
            additional = 3
        else:
            additional = 0

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=inchannels,out_channels=30,kernel_size=kernelsize,stride=1,padding=kernelsize//2, bias=True)
        self.conv2 = nn.Conv1d(in_channels=30,out_channels=50,kernel_size=kernelsize,stride=1,padding=kernelsize//2, bias=True)

        # fully connected layer and dropout
        self.fc0 = nn.Linear(in_features=(50*samplesPerChannel)+additional,out_features=amountFeatures, bias=True)
        self.fc1 = nn.Linear(in_features=amountFeatures,out_features=50, bias=True)
        self.fc2 = nn.Linear(in_features=50,out_features=2, bias=True)

        # Dropout enabled / disabled
        self.dropoutEnabled = dropoutEnabled
        self.debug = debug
        self.dropout = nn.Dropout(0.5)

    def forward(self,x,AoA):

        # Test if correct data is being inserted
        if(self.debug):
            plottingUtils.plotData(np.concatenate((x[0][0], x[0][2], x[0][4], x[0][6]),0), "Even indices",-1)
            plottingUtils.plotData(np.concatenate((x[0][1], x[0][3], x[0][5], x[0][7]),0), "Odd indices",-1)
            plottingUtils.plotData(x[0][1], "index 1",-1)
            plottingUtils.plotData(np.concatenate((np.sqrt(x[0][0]**2 + x[0][1]**2), np.sqrt(x[0][2]**2 + x[0][3]**2), 
                np.sqrt(x[0][4]**2 + x[0][5]**2), np.sqrt(x[0][6]**2 + x[0][7]**2)),0), "Abs distance with even/odd indices",-1)

        # Pass input from layer to layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Reshape it to one vector
        x = x.view(-1, self.num_flat_features(x))

        # AoA enabled -> add 3 values to the vector
        if(self.AoAenabled):
            x = torch.cat((x, AoA), dim=1)

        # Fully connected layer
        x = F.relu(self.fc0(x))

        # Discard some hidden units if dropout used
        if(self.dropoutEnabled):
            x = self.dropout(x)

        x = F.relu(self.fc1(x))        

        # For the last layer, ReLu is not used -> otherwise, the network would only produce positive estimates
        x = self.fc2(x)
        return x
    
    # Flat features => to fit them into the input layer of a linear layer
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# --------------------------------- ENABLE/DISABLE FEATURES ------------------------------------------------------ #

    # Enable and disable Dropout
    def enableDropout(self):
        self.dropoutEnabled = True

    def disableDropout(self):
        self.dropoutEnabled = False

    # Enable and disable Debugging
    def enableDebug(self):
        self.debug = True

    def disableDebug(self):
        self.debug = False

#----------------------------------------- LOSS FUNCTIONS ----------------------------------------------------------#

    # Distance loss function => for testing / verifying as it provides an intuitive understanding
    def criterionCartesianCordDistance(self,x,target):
        loss = 0
        for i in range(0,len(x)):     
            loss_crit = torch.sqrt((abs(x[i,0]-target[i,0]))**2 + (abs(x[i,1]-target[i,1]))**2)
            loss = loss + (1/len(x))*loss_crit
            if(self.debug):
                print(loss_crit)
        return loss

    # Different Criteria / Lossfunctions
    def criterionCartesianCord(self,x,target):
        loss = 0
        for i in range(0,len(x)):     
            loss_crit = ((abs(x[i,0]-target[i,0]))**2 + (abs(x[i,1]-target[i,1]))**2)**3
            loss = loss + (1/len(x))*loss_crit
            if(self.debug):
                print(loss_crit)
        return loss

    # Max distance is 180 degrees ... 0° and 355 are pretty close!!!
    # [r,a] -> radius and angle
    def criterionAngleDist(self,x,target):
        loss = 0
        for i in range(0,len(x)):
            radius_diff = abs(x[i,0] - target[i,0])
            temp = abs(x[i,1] - target[i,1]) % 360
            angle_diff = min(temp, 360-temp)
            loss_crit = 0.5 * torch.sqrt((radius_diff**2 + angle_diff**2))
            loss = loss + (1/len(x))*loss_crit
            if(self.debug):
                print("Radius difference of %.3f and angle difference of %.3f lead to a loss of: %.3f" % (radius_diff, angle_diff, loss_crit))
        return loss

#------------------------------------------- LEARNINGRATE ------------------------------------------------------------#    

    # Source: https://discuss.pytorch.org/t/adaptive-learning-rate/320
    # Adjust learning rate manually
    def adjust_learning_rate(self, optimizer, epoch, lr, updatefreq):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_new = lr * (0.1 ** (epoch // updatefreq))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new
        return lr_new
        
######################################################################################################################

# ------------------------------------------------ TRAINING -------------------------------------------------------- #

# Train the network
# Initializes a network with the passed parameters
# Prints debug information if debug == true
# Errors are added to the errorLog
def trainNetwork(data, numTraining, batchsize, droprate, kernelsize, inchannels,
    learningrate, momentum, countepochs, updatefreq, printfreq, lossfunction, normalizeData, 
    backprop, dropoutEnabled, insertAoA, amountFeatures, debug, errorLog):

    # Create trainingset and load it into a setloader
    if(len(data) != numTraining):
        utils.printFailure("Error happened in trainNetwork @ learning: length of data not equal to amount of trainingdata")
        errorLog.append("Error happened in trainNetwork @ learning: length of data not equal to amount of trainingdata")

    trainingSet = data

    # Includes normalization, if desired
    dataSetTraining = dataManagement.CSIDataset(trainingSet, normalizeData, -1)
    trainloader=torch.utils.data.DataLoader(dataSetTraining, batch_size=batchsize, shuffle=True, num_workers=4)

    samplesPerChannel = len(data[0][0][0])

    # Init network with given parameters
    net = Net(droprate,kernelsize,inchannels,samplesPerChannel, dropoutEnabled, debug, insertAoA,amountFeatures)

    # Create optimizer depending on the used backpropagation algorithm
    if(backprop == enumerators.BACKPROP.SGD):   
        optimizer = optim.SGD(net.parameters(),learningrate, momentum=momentum)
    elif(backprop == enumerators.BACKPROP.ADAM):
        optimizer = optim.Adam(net.parameters(),learningrate)

    print("Choosen batchsize: %d; learning rate: %.3f; Number of trainingsamples: %d (total samples: %d)" % (batchsize, learningrate, numTraining, len(data)))

    # Loop over the dataset multiple times -> amount of epochs
    for epoch in range(countepochs):
        
        lr_act = learningrate
        
        # Only adjusts, if SGD is used
        if(backprop == enumerators.BACKPROP.SGD):
            lr_act = net.adjust_learning_rate(optimizer, epoch, learningrate,updatefreq)
        
        print("Epoch number %d with learning rate %.3f and batchsize: %d" % (epoch+1, lr_act, batchsize))
        
        running_loss = 0.0

        # Iterate through the trainloader with a certain batchsize
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            # Add the AoA features if desired
            if(insertAoA):
                inputs, labels, AoA = data
            else:
                inputs, labels = data
                AoA = torch.tensor(-1.0) # not used

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float(), AoA.float())
            loss = -1
            
            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())


            if(debug):
                print("Actual lost: %.3f" % (loss))

            # Backpropagation
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % printfreq == printfreq - 1:    # print every printfreq mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / printfreq))
                running_loss = 0.0

    # Return the network and parameters for normalization
    return [net, [dataSetTraining.getMean(), dataSetTraining.getMin(), dataSetTraining.getMax()], errorLog]

# --------------------------------------------- REFINEMENT -------------------------------------------------------- #

# Refines an existing network
# As an old network is used, certain parameters are already contained in the old network (net)
def refineNetwork(data, net, params, batchsize, countepochs, printfreq, lossfunction,loadModelAsNet, learningrate,
    normalizeData, backprop, insertAoA, debug, errorLog):

    printfreq = (len(data)//batchsize)//2

    # Freeze all layers / weights
    for param in net.parameters():
        param.requires_grad= False

    # Get amount of input-features in the fully connected layers
    num_ftrs2 = net.fc2.in_features
    num_ftrs1 = net.fc1.in_features
    num_ftrs0 = net.fc0.in_features
    print("fc1: %d; fc2: %d" % (num_ftrs1,num_ftrs2))

    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTraining = dataManagement.CSIDataset(data, normalizeData, params)
    trainloader=torch.utils.data.DataLoader(dataSetTraining, batch_size=batchsize, shuffle=True, num_workers=4)

    # Initialize new layer to train them with the data (not frozen anymore)
    net.fc2 = nn.Linear(in_features=num_ftrs2,out_features=2, bias=True)

    net.fc1 = nn.Linear(in_features=num_ftrs1, out_features=num_ftrs2, bias=True)

    net.fc0 = nn.Linear(in_features=num_ftrs0, out_features=num_ftrs1, bias=True)

    # Print the architecture
    print(net)

    # create optimizer depending on the used backpropagation algorithm
    if(backprop == enumerators.BACKPROP.SGD):   # SGD
        optimizer = optim.SGD(net.parameters(),learningrate, momentum=momentum)
    elif(backprop == enumerators.BACKPROP.ADAM):
        optimizer = optim.Adam(net.parameters(),learningrate)

    print("Choosen batchsize: %d; learning rate: %.3f; Number of trainingsamples: %d (total samples: %d)" % (batchsize, learningrate, len(data), len(data)))

    for epoch in range(countepochs):  # loop over the dataset multiple times
        
        lr_act = learningrate
        
        print("Epoch number %d with learning rate %.3f and batchsize: %d" % (epoch+1, lr_act, batchsize))
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            # Add the AoA features if desired
            if(insertAoA):
                inputs, labels, AoA = data
            else:
                inputs, labels = data
                AoA = torch.tensor(-1.0) # not used

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float(), AoA.float())
            loss = -1
            
            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())


            if(debug):
                print("Actual lost: %.3f" % (loss))

            # Backpropagation
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % printfreq == printfreq - 1:    # print every printfreq mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / printfreq))
                running_loss = 0.0

    return [net, params, errorLog]

# ------------------------------------------------ TESTING -------------------------------------------------------- #

# Test an existing network
# If training data is used -> validity check over number of trainingsamples
def testNetwork(data, net, numTraining, batchsize, lossfunction, useTrainingData, normalizeData, params, insertAoA, debug, errorLog):

    if(debug):
        net.enableDebug()
    else:
        net.disableDebug()

    # Create testset/trainingset and load it into a setloader to get mean error
    if(useTrainingData and len(data) != numTraining):
        utils.printFailure("Error happened in testNetwork @ learning: length of data not equal to amount of trainingdata")
        errorLog.append("Error happened in testNetwork @ learning: length of data not equal to amount of trainingdata")


    testSet = data

    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTest = dataManagement.CSIDataset(testSet, normalizeData, params)
    testloader=torch.utils.data.DataLoader(dataSetTest, batch_size=batchsize, shuffle=False, num_workers=4)

    total_loss = 0
    with torch.no_grad():
        for entry in testloader:

            # Add the AoA features if desired
            if(insertAoA):
                samples, labels, AoA = entry
            else:
                samples, labels = entry
                AoA = torch.tensor(-1.0) # not used

            # convert input to float and feed it into the network to get the output
            outputs = net(samples.float(), AoA.float())
            loss = -1

            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())
            
            if(debug):
                print("Actual lost: %.3f" % (loss))

            total_loss = total_loss + loss

    # Calculate mean error
    mean_error = total_loss / len(testloader)
    return mean_error, errorLog

# ------------------------------------------- GET EXTREMES ------------------------------------------------------- #

# Method finds the top "numResults". If bestResults == TRUE -> Find best ones, if not: worst ones (=biggest loss)
def getExtremeEstimates(data, net, numTraining, lossfunction, numResult, bestResults, normalizeData, params, useTrainingData, insertAoA, debug, room, errorLog):
    
    if(debug):
        net.enableDebug()
    else:
        net.disableDebug()

    estimatedIndices = [[sys.maxsize,-1],[sys.maxsize,-1]]

    # Initialize the value with the best/worst possible
    initialval = -1
    if(bestResults):
        initialval = sys.maxsize
    
    # Initialize structure to store the top XX values
    result = []
    for i in range(0,numResult):
        result.append([initialval, -1])

    # Initialize Testset/Trainingsset to search for the top/worst values
    if(useTrainingData and len(data) != numTraining):
        utils.printFailure("Error happened in getExtremeEstimates @ learning: length of data not equal to amount of trainingdata")
        errorLog.append("Error happened in getExtremeEstimates @ learning: length of data not equal to amount of trainingdata")

    testSet = data
    
    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTest = dataManagement.CSIDataset(testSet, normalizeData, params)
    testloader=torch.utils.data.DataLoader(dataSetTest, batch_size=1, shuffle=False, num_workers=4)

    # Try every sample and collect the best/worst ones
    total_loss = 0
    with torch.no_grad():
        for i in range(0,len(testloader)):

            # Add the AoA features if desired
            if(insertAoA):
                samples, labels, AoA = dataSetTest[i]
            else:
                samples, labels = dataSetTest[i]
                AoA = np.array([-1.0]) # not used

            # Only pass one sample, but as 1-D array with one entry -> as a batchsize with only one sample is passed
            sample = torch.zeros(1,len(samples), len(samples[0]))
            inputAoA = torch.zeros(1,3)

            sample[0] = samples
            
            inputAoA[0] = torch.from_numpy(AoA)

            # convert input to float and feed it into the network to get the output
            outputs = net(sample.float(), inputAoA.float())
            loss = -1
            output = torch.zeros(1,2)
            label = torch.zeros(1,2)
            output[0] = outputs
            label[0] = labels
            
            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(output.float(), label.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(output.float(), label.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(output.float(), label.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(output.float(), label.float())

            if(not isinstance(loss, float)):
                loss = loss.item()

            # Compare actual item with the best/worst ones so far and sort the list again (sort the new value in the correct place)
            if(bestResults):
                if(loss < result[numResult-1][0]):
                    result[numResult-1] = [loss, i]
                    result = sorted(result)
            else:
                if(loss > result[numResult-1][0]):
                    result[numResult-1] = [loss, i]
                    result = sorted(result, reverse=True)

            outputs = utils.changeCoordRef(outputs[0], len(room))

            # Get indices range for estimated position => outputs 
            if(outputs[0].item() < estimatedIndices[0][0]):
                estimatedIndices[0][0] = outputs[0].item()
            if(outputs[1].item()< estimatedIndices[1][0]):
                estimatedIndices[1][0] = outputs[1].item()
            if(outputs[0].item() > estimatedIndices[0][1]):
                estimatedIndices[0][1] = outputs[0].item()
            if(outputs[1].item() > estimatedIndices[1][1]):
                estimatedIndices[1][1] = outputs[1].item()

    return result, estimatedIndices, errorLog

# --------------------------------------- CREATE SORTED LIST ------------------------------------------------------- #

# Create a sorted list; list of losses (feed input in neural network; compare output with label via lossfunction)
def createSortedList(data, net, numTraining, lossfunction, useTrainingData, normalizeData, params, insertAoA, debug, errorLog):
    
    if(debug):
        net.enableDebug()
    else:
        net.disableDebug()

    # Sanity check
    if(useTrainingData and len(data) != numTraining):
        utils.printFailure("Error happened in createSortedList @ learning: length of data not equal to amount of trainingdata")
        errorLog.append("Error happened in createSortedList @ learning: length of data not equal to amount of trainingdata")

    testSet = data
    
    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTest = dataManagement.CSIDataset(testSet, normalizeData, params)
    testloader=torch.utils.data.DataLoader(dataSetTest, batch_size=1, shuffle=False, num_workers=4)

    sortedOutputs = np.array([])

    with torch.no_grad():

        # Compute loss for every sample (only one sample per time, as batchsize = 1)
        for entry in testloader:

            # Add the AoA features if desired
            if(insertAoA):
                samples, labels, AoA = entry
            else:
                samples, labels = entry
                AoA = torch.tensor(-1.0) # not used

            outputs = net(samples.float(), AoA.float())

            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())
            
            if(debug):
                print("Actual lost: %.3f" % (loss))

            # store result / append it
            sortedOutputs = np.append(sortedOutputs,loss)

    # Sort array
    sortedOutputs.sort(0)

    return sortedOutputs, errorLog

# --------------------------------------- CREATE HEAT MAP -------------------------------------------------------- #

# Creates a heatmap of a given network. While green positions indicate good estimates, red ones represent bad ones
def createHeatMap(data, net, lossfunction, amountOfPos, params, insertAoA, debug, room, excluded, normalizeData):

    if(debug):
        net.enableDebug()
    else:
        net.disableDebug()
 
    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTest = dataManagement.CSIDataset(data, normalizeData, params)
    testloader=torch.utils.data.DataLoader(dataSetTest, batch_size=1, shuffle=False, num_workers=4)

    # Store the error as well as the amount of the estimates per position
    heatMap_Error = dict.fromkeys(np.arange(1,amountOfPos+1), 0.0)
    heatMap_Samples = dict.fromkeys(np.arange(1,amountOfPos+1), 0.0)

    print("Before: %d" % (len(heatMap_Error)))

    # delete the keys of excluded positions
    if isinstance(excluded,set):
        for entry in excluded:
            del heatMap_Error[entry]
            del heatMap_Samples[entry]

    print("After: %d" % (len(heatMap_Error)))

    # Creates dataManagement object, which provides data operations 
    processor = dataManagement.DataProcessing(debug)

    # initializes the values to store the current best/ worst
    best = 100000
    worst = 0

    with torch.no_grad():
        for entry in testloader:

            # Add the AoA features if desired
            if(insertAoA):
                samples, labels, AoA = entry
            else:
                samples, labels = entry
                AoA = torch.tensor(-1.0) # not used

            # Calculate the outputs
            outputs = net(samples.float(), AoA.float())

            # Get the index in the grid corresponding to the coordinate
            number = processor.getNumberOfPos(labels[0].numpy(), room)

            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())
            
            # Update best and worst
            if(loss > worst):
                worst = loss

            if(loss < best):
                best = loss

            # Add the error to the correct position and increase the amount of measurements for this position
            heatMap_Error[number] = heatMap_Error[number] + loss
            heatMap_Samples[number] = heatMap_Samples[number] + 1

            if(debug):
                print("Actual lost: %.3f" % (loss))

    if(debug):
        print(heatMap_Error)

    # Calculate the mean error for every position
    for key in heatMap_Error:
        heatMap_Error[key] = heatMap_Error[key] / heatMap_Samples[key]

    if(debug):
        print(heatMap_Error)

    return [heatMap_Error, [best,worst]]

# ------------------------------------------- CREATE HITMAP ------------------------------------------------------- #

# Creates a hitmap of a given network. The darkness of the position indicates the amount of hits on the respective field.
# If the parameter forAll is set, the hits for all different positions are collected and displayed in one plot
# Else: The estimates are grouped after the true location (shown in green)
def createHitMap(data, net, lossfunction, amountOfPos, params, insertAoA, debug, room, excluded, normalizeData, forAll):
    
    if(debug):
        net.enableDebug()
    else:
        net.disableDebug()
 
    # Normalize data if desired (with passed parameters -> normalized the same way as before)
    dataSetTest = dataManagement.CSIDataset(data, normalizeData, params)
    testloader=torch.utils.data.DataLoader(dataSetTest, batch_size=1, shuffle=False, num_workers=4)

    if forAll:
        hitmap = {"1":[np.zeros(room.shape), 0, 0]}
    else:
        # create a dictionary for all positions
        hitmap = dict.fromkeys(np.arange(1,amountOfPos+1), [])

        # initialize all positions with empty rooms
        for key in hitmap:
            hitmap[key] = [np.zeros(room.shape), 0, 0]

        # Delete the positions, which are excluded
        if isinstance(excluded,set):
            for entry in excluded:
                del hitmap[entry]

    # Creates dataManagement object, which provides data operations
    processor = dataManagement.DataProcessing(debug)

    # initializes the values to store the current best/ worst
    best = 100000
    worst = 0

    with torch.no_grad():
        for entry in testloader:

            # Add the AoA features if desired
            if(insertAoA):
                samples, labels, AoA = entry
            else:
                samples, labels = entry
                AoA = torch.tensor(-1.0) # not used

            # Calculate the outputs
            outputs = net(samples.float(), AoA.float())

            if forAll:
                # Always same index, as all hits are collected independent of the correct position
                number = "1"
            else:    
                # Get the index in the grid corresponding to the coordinate of the correct position (for which the hits were estimated)
                number = processor.getNumberOfPos(labels[0].numpy(), room)

            # Calculate loss depending on used lossfunction
            if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
                loss = net.criterionCartesianCord(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
                loss = net.criterionAngleDistA(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
                criterion = nn.MSELoss()
                loss = criterion(outputs.float(), labels.float())
            elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
                loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())
            
            # Update best and worst
            if(loss > worst):
                worst = loss

            if(loss < best):
                best = loss

            # For the correct index, the amount of hits is increased for the certain position
            if(outputs[0].numpy() < hitmap[number][0].shape).all():
                hitmap[number][0][int(outputs[0][0])][int(outputs[0][1])] = hitmap[number][0][int(outputs[0][0])][int(outputs[0][1])] + 1
                hitmap[number][1] = hitmap[number][1] + 1
            else:
                # If the position is outside of the grid, the outlier are increased
                hitmap[number][2] = hitmap[number][2] + 1

            if(debug):
                print("Actual lost: %.3f" % (loss))

    return hitmap

# ------------------------------------------- LOSS FUNCTION ------------------------------------------------------- #

# Get loss for a certain lossfunction for given parameters
def getLoss(lossfunction, output, label):

    # define dummy net, only used to get Loss
    net = Net(0, 1, 8, 256, False, False, False,10)

    outputs = torch.zeros(1,2)
    labels = torch.zeros(1,2)
    outputs[0] = output
    labels[0] = label
    
    # Calculate loss depending on used lossfunction
    if(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN):
        loss = net.criterionCartesianCord(outputs.float(), labels.float())
    elif(lossfunction == enumerators.LOSSFUNCTIONS.ANGLEDIST):
        loss = net.criterionAngleDistA(outputs.float(), labels.float())
    elif(lossfunction == enumerators.LOSSFUNCTIONS.MSELOSS):
        criterion = nn.MSELoss()
        loss = criterion(outputs.float(), labels.float())
    elif(lossfunction == enumerators.LOSSFUNCTIONS.CARTESIAN_DIST):
        loss = net.criterionCartesianCordDistance(outputs.float(), labels.float())

    return loss

#####################################################################################################################