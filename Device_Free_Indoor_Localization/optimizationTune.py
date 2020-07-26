#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Kosterhon
"""

# Call this method for hyperparameter optimization and scalable training

# This class provides the training of models together with hyperparameter search
# The framework TUNE is used to do this
# The code is based on their tutorial, which can be found at:
# https://ray.readthedocs.io/en/latest/tune-usage.html

# IMPORTANT: Instead of using the Tune Logging mechanisms, an own one is used

# ---------------------------------------------------- IMPORTS --------------------------------------------------------------- #

import essentialMethods
import parameters
import enumerators
import learning
import dataManagement
import utils

import os
from random import randint

import torch
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ray
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

# --------------------------------------------------------------------------------------------------------------------------- #

# This splitNo depends on the location, where it is stored!
# print it to adapt it, just used to identify the model
# -> for the tested environments, it was set to 4 or 5
splitNo = 5

# ---------------------------------------------------- IMPORTANT PARAMETERS ------------------------------------------------- #

# Change these values if you want the training to run quicker or slower.
# Terminate the training/testing after this amount of samples
EPOCH_SIZE = 4096  # -> Batchsize * Number of iterations (batches) <= EPOCH / TEST SIZE
TEST_SIZE = 512

# Stopping Criteria:
STOP_ITERATIONS = 400    

STOP_MEAN_LOSS = 2.6	# 2.6 squares -> 0.78m

STOP_TIME = 7200

NO_SAMPLES = 5

NUM_CPUS = 1   

# --------------------------------------- EARLY STOPPING / COSTUM STOPPING CONDITIONS --------------------------------------- #

# Checks, if one of the stopping criteria is reached
def stopFunct(trial_id, result):
        if (result["training_iteration"] >= STOP_ITERATIONS
            or result["time_total_s"] >= STOP_TIME or result["mean_loss"] <= STOP_MEAN_LOSS):
            return True
        return False

# ----------------------------------------- TRAINABLE PASSIVE LOCALIZATION FRAMEWORK ---------------------------------------- #

# Trainable class for localization
class passiveLocalization(tune.Trainable):

	# Setup of Trainable 
    def _setup(self, config):

        # Set the dynamic parameters, which should be optimized
        self.amountFeatures = int(config["amountFeatures"])
        self.batchsize = int(config["batchsize"])

    	# Use GPUs if possible
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Get the train- and testloader as well as the normalization-parameters
        self.train_loader, self.test_loader, self.params = get_data_loaders(parameters.batchsize)

        # Initialize the network
        self.model = learning.Net(parameters.droprate, parameters.kernelsize, parameters.inchannels, parameters.samplesPerChannel,
						parameters.dropoutEnabled, False, 
						parameters.insertAoAInNetwork, int(config["amountFeatures"])).to(self.device)

        # Initialize Adam Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),parameters.learningrate)

        # Initialize LogString
        self.logString = "empty"

    # train the network
    def _train(self):

    	# train network
        train(
            self.model, self.optimizer, self.train_loader, device=self.device)

        # evaluate loss
        loss = test(self.model, self.test_loader, self.device)
        
        return {"mean_loss": loss}

    # save the model
    def _save(self, checkpoint_dir):
        
        # create model name
        model_name = "model_%s" % (''.join(checkpoint_dir.split('/')[splitNo].split('_')[2:4]))
        
        # Default code
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")

        config = {"insertAoAInNetwork":parameters.insertAoAInNetwork, "selectFromSamplesPerPos":parameters.selectFromSamplesPerPos, "mask":parameters.mask,
                "shuffling":parameters.shuffling, "randomDigest":parameters.randomDigest, "params":self.params,
                "filterDataAfterLoading":parameters.filterDataAfterLoading, "excludedArea":parameters.excludedArea,
                "numTraining":parameters.numTraining, "normalizeData":parameters.normalizeData, "amountFeatures":self.amountFeatures, "antennasRXUsed":parameters.antennasRXUsed, 
                "antennasTXUsed":parameters.antennasTXUsed, "samplesPerChannel":parameters.samplesPerChannel}

        # Save the model via utils as well as via tune framework (torch.save)
        utils.saveFile(model_name,[self.model,config])
        torch.save(self.model.state_dict(), checkpoint_path)

        # Embed the name of the model in the resulting log String
        string = self.logString.split('<>')[0] + model_name + self.logString.split('<>')[2][len(model_name)-5:]

    	# Open File with "a" => append a log entry to the logfile
        f = open(parameters.tuneLogFile, "a")
        f.write(string)
        f.close()
        
        return checkpoint_path

    # Restoring: default method
    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    # Log the result
    def _log_result(self, result):
        print("Log called")

    	# Now create a log string and store it in the class
        self.logString = utils.getLogString("<> <>", parameters.logFile, parameters.netID, parameters.dataID, parameters.lossID, 
    			result["date"], result["time_total_s"], parameters.learningrate, parameters.momentum, parameters.batchsize, 
    			parameters.droprate, parameters.kernelsize, parameters.inchannels, result["training_iteration"], parameters.lossfunctionTraining, 
    			parameters.lossfunctionTesting, parameters.numTraining, EPOCH_SIZE, TEST_SIZE, parameters.insertAoAInNetwork, 
    			result["mean_loss"], parameters.backprop, parameters.logWidth, parameters.dropoutEnabled, parameters.shuffling, 
    			parameters.selectFromSamplesPerPos, parameters.normalizeData, parameters.usePrevRandomVal, parameters.filterDataAfterLoading, 
    			parameters.randomDigest, parameters.mask, parameters.excluded, parameters.exceptRow, parameters.exceptCol, parameters.useCombinedData, 
    			parameters.baseNo, self.amountFeatures, parameters.numValidation) 


# ---------------------------------------- INTERFACE METHODS ---------------------------------------------------------------- #

# Training Interface
def train(model, optimizer, train_loader, device=torch.device("cpu")):

	model.train()

	# For each batch-iteration with one mini-batch of data
	for batch_idx, data in enumerate(train_loader):

        # get the inputs; data is a list of [inputs, labels, (optional AoA)]
		if(parameters.insertAoAInNetwork):
			inputs, labels, AoA = data
		else:
			inputs, labels = data
			AoA = torch.tensor(-1) # not used; dummy value for AoA	

		# check termination criteria
		if batch_idx * len(inputs) > EPOCH_SIZE:
			return

		inputs, labels, AoA = inputs.to(device), labels.to(device), AoA.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		output = model(inputs.float(), AoA.float())

		# Calculate loss depending on used lossfunction: Set it by hand -> better runtime behaviour
		loss = model.criterionCartesianCordDistance(output.float(), labels.float())

		# Backpropagation
		loss.backward()
		optimizer.step()

# Testing Interface
def test(model, data_loader, device=torch.device("cpu")):
	
	model.eval()

	# Initialize Loss and Counter
	total_loss = 0
	counter = 0

	with torch.no_grad():

		# For each batch-iteration with one mini-batch of data
		for batch_idx, entry in enumerate(data_loader):

			# get the inputs; data is a list of [inputs, labels, (optional AoA)]
			if(parameters.insertAoAInNetwork):
			    inputs, labels, AoA = entry
			else:
			    inputs, labels = entry
			    AoA = torch.tensor(-1) # not used

			# check termination criteria
			if batch_idx * len(inputs) > TEST_SIZE:
				break

			inputs, labels, AoA = inputs.to(device), labels.to(device), AoA.to(device)

			# Calculate the output of the network
			outputs = model(inputs.float(), AoA.float())
			loss = -1

			# Calculate loss depending on used lossfunction => hardcoded, as should be always the same
			loss = model.criterionCartesianCordDistance(outputs.float(), labels.float())

			# Store result and increment counter
			total_loss = total_loss + loss
			counter = counter + 1


	# Calculate mean error
	mean_error = total_loss / counter

	return mean_error


# Get the training - and testing set as well as the normalization parameter
def get_data_loaders(batchsize):

    # Load data for Training
    trainingSet, trainingSetLen, _ = essentialMethods.loadData(True, enumerators.DATA.TRAINING, True, [])

    # Create Trainingloader
    dataSetTraining = dataManagement.CSIDataset(trainingSet, parameters.normalizeData, -1)
    train_loader=torch.utils.data.DataLoader(dataSetTraining, batch_size=batchsize, shuffle=True)

    # Load data for Testing (not shuffled -> only the first xx data is used)
    testSet, testSetLen, _ = essentialMethods.loadData(False, enumerators.DATA.TESTING, True, [])

    # Params: to normalize the testdata the same way as the trainingdata -> no information leakage
    params = [dataSetTraining.getMean(), dataSetTraining.getMin(), dataSetTraining.getMax()]

    # Constrain TestSet to ValidationSet
    testSet = testSet[:parameters.numValidation]

    # Create Testloader 
    dataSetTest = dataManagement.CSIDataset(testSet, parameters.normalizeData, params)
    test_loader=torch.utils.data.DataLoader(dataSetTest, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader, params

# ------------------------------------------------------- MAIN CALL --------------------------------------------------------- #


if __name__ == "__main__":

	# Restrict CPU and GPU Usage
    ray.init(num_cpus=NUM_CPUS, num_gpus=0)

    # Searchspace
    space = {"amountFeatures":(20,500), "batchsize":(20, 200)}

    # Bayesian Search for Hyperparameter optimization
    algo = BayesOptSearch(
    	space,
    	random_state=randint(0,10000),
    	max_concurrent=NUM_CPUS,
    	metric="mean_loss",
    	mode="min",
    	utility_kwargs={	
    		"kind": "ucb",
    		"kappa": 2.5,
    		"xi": 0.0
    	})

    # Start the optimizer
    analysis = tune.run(
        passiveLocalization,
        name="trainPassiveLocalizationNetwork",
        search_alg=algo,
        stop=stopFunct,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        num_samples= NO_SAMPLES,
        checkpoint_at_end=True,
        checkpoint_freq=1)
