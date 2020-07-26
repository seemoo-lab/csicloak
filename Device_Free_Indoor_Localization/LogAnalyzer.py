#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides methods to search in the created logs
-> independent of all other files!

"""

#########################################################################################################

import sys
from enum import Enum
from tabulate import tabulate

######################################## ENUM ###########################################################

class MODE(Enum):
    DATA = 0		# contains Baseloader and Coaxloader!
    TRAINING = 1
    TESTING = 2
    WORST = 3
    BEST = 4
    BASELINE = 5
    BASE_DATA_COMB = 6
    NONE = 7
    CDF = 8
    AOA_OF_DATA = 9

##################################### Parameters ########################################################

# Interesting files:
directory ="/Path/To/sampleFiles/logs/" 

# Naming Conventions so far:
# - evalCDF_* 		-> results of the ANALYZETUNEMODELS method
# - tuneResults_* 	-> logging during the training, contains all intermediate values 
# - none of it:		-> regular log file

# Sample logs:
sampleFiles =  ["evalCDF_example.log", "example.log", "tuneResults_example.log"]

file = sampleFiles[0]

# Determine, which information should be printed
printSections = True
printAll = False
printTables = True

# Sort in descending or ascending order
descending = False 

# ---------------------------------------------------------------------------------------------------- #

if "tuneResults" in file:
	analyzeTuneLog = True
	printSections = False
	crit = "Epochs"
else:
	analyzeTuneLog = False
	crit = "MeanTrainingError"

if "evalCDF" in file:
	analyzeTuneTraining = True 
	printSections = False
else:
	analyzeTuneTraining = False

sortCrit = {"DATA": "DataID", "TRAINING": crit, "TESTING": "MeanTestingError", "WORST":"Error range from", "BEST": "To", 
			"BASELINE": "DataID", "BASE_DATA_COMB": "DataID", "CDF": "Median", "BASELINE":"DataID", "AOA_OF_DATA": "DataID"}

act_mode = MODE.NONE

if "tuneResults" in file:
	categoriesToHide = {"Backprop", "Learningrate", "Kernelsize", "Momentum", "Inchannels", "Droprate activated", "Droprate", "Found in columns", 
					"Found in rows", "Error using Mean of Training Labels (Training)", "Error using Mean of Used Room (Training)", 
					"Data filtered afterwards", "Shuffling", "Normalization","Testfunction",
					"Datadirectory", "Basedirectory", "Random Permutation", "Trainingfunction", "LossID",
					"Hidden units", "Epochsize", "Select from x Samples per position", "Testsize", "Aoa as a feature", "NetID", "Amount of Trainingsamples", "Batchsize", "ID","Aoa as a feature"}
else:
	categoriesToHide = {"Backprop", "Learningrate", "Kernelsize", "Momentum", "Inchannels", "Droprate activated", "Droprate", "Found in columns", 
					"Found in rows", "Error using Mean of Training Labels (Training)", "Error using Mean of Used Room (Training)", 
					"Data filtered afterwards", "Shuffling", "Normalization","Testfunction",
					"Datadirectory", "Basedirectory", "Random Permutation", "Trainingfunction", "LossID"} #"Aoa as a feature"

#########################################################################################################

resetInformation = {"DataID": "xxxx", "LossID": "xxxx", "NetID": "xxxx", "Backprop": "xxxx", 
					"Learningrate": "xxxx", "Momentum": "xxxx", "Batchsize": "xxxx", "Droprate": "xxxx",
					"Kernelsize": "xxxx", "Epochs": "xxxx", "Trainingfunction": "xxxx", "Testfunction": "xxxx",
					"Amount of Trainingsamples": "xxxx", "MeanTrainingError": "xxxx", "MeanTestingError": "xxxx",
					"Amount of Testsamples": "xxxx", "Size of Top-List": "xxxx", "Found in columns": "xxxx", "Found in rows": "xxxx",
					"Error range from": "xxxx", "To": "xxxx", "Amount of Samples": "xxxx", 
					"Coordinate Format": "xxxx", "Signal Representation": "xxxx", "Room": "xxxx", 
					"First Quartile":"xxxx","Median":"xxxx","Third Quartile":"xxxx", "ID": "xxxx",
					"Error using Mean of Training Labels (Training)": "xxxx", "Error using Mean of Used Room (Training)": "xxxx",
					"Error using Mean of Training Labels (Testing)": "xxxx", "Error using Mean of Used Room (Testing)": "xxxx",
					"Data was combined with sample":"xxxx", "Select from x Samples per position": "xxxx",
					"Random Permutation": "xxxx", "Shuffling": "xxxx", "Aoa as a feature":"xxxx",
					"Normalization":"xxxx", "Droprate activated":"xxxx", "Datadirectory":"xxxx",
					"Saved As":"xxxx", "Data filtered afterwards": "xxxx", "Searched Set":"xxxx",
					"Basedirectory":"xxxx", "Data-Area":"xxxx","Hidden units":"xxxx", "Epochsize":"xxxx",
					"Testsize":"xxxx"}

actInformation = dict(resetInformation)

def resetActInformation():
	global resetInformation
	global actInformation

	actInformation = dict(resetInformation)

def printIt(property, output="default"):
	if(actInformation[property] != "xxxx"):
		if(output == "default"):
			output = property
		print("%s: %s" % (output, actInformation[property]))

def printInformation():
	global actInformation

	print("--------------------------")
	print("Mode: %s" % (act_mode))
	print("--------------------------")

	for key in actInformation:
		printIt(key)

def prepareToSort(actInformation):

	toRemove = []

	for key in actInformation:
		if(actInformation[key] == "xxxx"):
			toRemove.append(key)
		
	for key in toRemove:
		del actInformation[key]

	return actInformation

def sortStuff(val):
	return val[sortCrit[act_mode]]

def sortStuff2(val):
	return int(val["EpochsTrained"])

def hideCols(tableData):
	for category in categoriesToHide:
		if category in tableData.keys():
			del tableData[category]

	return tableData

def getLogContent(file, directory):

	print(file)
	print(directory)

	logContent = {"DATA":[], "AOA_OF_DATA": [], "BASE_DATA_COMB": [], "TRAINING":[], "TESTING":[], "WORST":[], "BEST":[], "BASELINE": [], "CDF": []}

	global act_mode
	act_mode = MODE.NONE

	markerDetected = False
	observedUsedRoom = False
	observedOutputMean = False

	id_ctr = 0
	id_act = 0

	linecount = 0
	with open(directory+file, encoding= 'utf-8') as f:
		for line in f:

			# New mode begins
			if markerDetected and act_mode == MODE.NONE:
				if "BASE APPLIED TO DATA" in line:
					act_mode = MODE.BASE_DATA_COMB
				elif "GET AOA OF DATA" in line:
					act_mode = MODE.AOA_OF_DATA
				elif "TRAINING" in line:
					act_mode = MODE.TRAINING
					actInformation["ID"] = id_ctr
					id_act = id_ctr
					id_ctr= id_ctr + 1
				elif "TESTING" in line:
					act_mode = MODE.TESTING
					actInformation["ID"] = id_act
				elif "WORST" in line:
					act_mode = MODE.WORST
					actInformation["ID"] = id_act
					actInformation["Size of Top-List"] = line[1:-2].strip().split(' ')[1]
				elif "BEST" in line:
					act_mode = MODE.BEST
					actInformation["ID"] = id_act
					actInformation["Size of Top-List"] = line[1:-2].strip().split(' ')[1]
				elif "BASELINE" in line:
					act_mode = MODE.BASELINE
					actInformation["ID"] = id_act
				elif "DATA" in line:
					act_mode = MODE.DATA
				elif "CDF" in line:
					act_mode = MODE.CDF
					actInformation["ID"] = id_act
				else:
					print("SEPARATION MISTAKE")

				#print(act_mode)
			elif markerDetected:
				# Save all the settings
				logContent[act_mode.name].append(prepareToSort(actInformation))

				if(printAll):
					printInformation()
				resetActInformation()
				act_mode = MODE.NONE


			elif "netID" in line:
				actInformation["NetID"] = line.split(' ')[2]
				actInformation["DataID"] = line.split(' ')[6]
				actInformation["LossID"] = line.split(' ')[4]
			elif "dataID" in line:
				actInformation["DataID"] = line.split(' ')[2]
			elif "backprop" in line: 
				actInformation["Backprop"] = line.split(' ')[3]
			elif "learningrate" in line:
				actInformation["Learningrate"] = line.split(' ')[2][:-1]
				actInformation["Momentum"] = line.split(' ')[4][:-1]
				actInformation["Batchsize"] = line.split(' ')[6]
			elif "droprate" in line:
				actInformation["Droprate"] = line.split()[2]
				actInformation["Droprate activated"] = line.split()[4][:-1]
			elif "kernelsize" in line:
				actInformation["Kernelsize"] = line.split(' ')[2][:-1]
				actInformation["Inchannels"] = line.split(' ')[4][:-1]
			elif "epochs" in line:
				actInformation["Epochs"] = int(line.split(' ')[2])
			elif "lossfunction used for training" in line:
				actInformation["Trainingfunction"] = line.split(' ')[5]
			elif "lossfunction used for testing" in line:
				actInformation["Testfunction"] = line.split(' ')[5]
			elif "lossfunction" in line and (act_mode == MODE.TESTING or act_mode == MODE.CDF or act_mode == MODE.BASELINE or act_mode == MODE.BEST or act_mode == MODE.WORST):
				actInformation["Testfunction"] = line.split(' ')[2]
			elif "Achieved mean training error" in line and observedUsedRoom and act_mode == MODE.BASELINE:
				actInformation["Error using Mean of Used Room (Training)"] = line.split(' ')[5]
			elif "Achieved mean testing error" in line and observedUsedRoom and act_mode == MODE.BASELINE:
				actInformation["Error using Mean of Used Room (Testing)"] = line.split(' ')[5]
			elif ("mean training error" in line and not act_mode == MODE.BASELINE) or ("mean testing error" in line and act_mode == MODE.TRAINING):
				actInformation["MeanTrainingError"] = line.split(' ')[5]
			elif "mean testing error" in line and not act_mode == MODE.BASELINE:
				actInformation["MeanTestingError"] = line.split(' ')[5]
			elif "errors in the range from" in line:
				actInformation["Error range from"] = line.split(' ')[7]
				actInformation["To"] = line.split(' ')[9]
			elif "indices found in col" in line:
				actInformation["Found in columns"] = line.split(' ')[6]
			elif "indices found in row" in line:
				actInformation["Found in rows"] = line.split(' ')[6]
			elif "amount of test samples" in line:
				actInformation["Amount of Testsamples"] = line.split(' ')[5]
			elif "amount of samples used for training" in line:
				actInformation["Amount of Trainingsamples"] = line.split(' ')[7]
			elif "Amount of samples" in line and (act_mode == MODE.DATA or act_mode == MODE.AOA_OF_DATA or act_mode == MODE.BASE_DATA_COMB):
				actInformation["Amount of Samples"] = line.split(' ')[4]
			elif "signal repr" in line:
				actInformation["Signal Representation"] = line.split(' ')[3]
			elif "coordinates format" in line:
				actInformation["Coordinate Format"] = line.split(' ')[3]
			elif "first quartile" in line:
				actInformation["First Quartile"] = line.split(' ')[5]
			elif "third quartile" in line:
				actInformation["Third Quartile"] = line.split(' ')[5]
			elif "median" in line:
				actInformation["Median"] = line.split(' ')[7]
			elif "room" in line and (act_mode == MODE.DATA or act_mode == MODE.BASE_DATA_COMB or act_mode == MODE.AOA_OF_DATA):
				actInformation["Room"] = line.split(' ')[3]
			elif "Achieved mean training error" in line and observedOutputMean and act_mode == MODE.BASELINE:
				actInformation["Error using Mean of Training Labels (Training)"] = line.split(' ')[5]
			elif "Achieved mean testing error" in line and observedOutputMean and act_mode == MODE.BASELINE:
				actInformation["Error using Mean of Training Labels (Testing)"] = line.split(' ')[5]
			elif "amount of training samples" in line:
				actInformation["Amount of Trainingsamples"] = line.split(' ')[5]
			elif "data was combined with" in line:
				actInformation["Data was combined with sample"] = line.split(' ')[9]
			elif "data directory" in line or ("directory" in line and act_mode == MODE.DATA):
				actInformation["Datadirectory"] = line.split(' ')[3]
			elif "AoA used as extra feature" in line:
				actInformation["Aoa as a feature"] = line.split(' ')[6]
			elif "Selected from" in line:
				actInformation["Select from x Samples per position"] = line.split(' ')[3]
			elif "Normalization" in line:
				actInformation["Normalization"] = line.split()[2]
			elif "Shuffling" in line:
				actInformation["Shuffling"] = line.split(' ')[3]
			elif("Random permutation") in line:
				actInformation["Random Permutation"] = line.split(' ')[3]
			elif "saved as" in line:
				actInformation["Saved As"] = line.split(' ')[3]
			elif "Epochsize" in line:
				actInformation["Epochsize"] = line.split(' ')[2][:-1]
				actInformation["Testsize"] = line.split(' ')[4]
			elif "Hidden units" in line:
				actInformation["Hidden units"] = line.split(' ')[8]
			elif "filtered after Loading" in line:
				actInformation["Data filtered afterwards"] = line.split(' ')[6]
			elif "Searched in the" in line:
				actInformation["Searched Set"] = line.split(' ')[4]
			elif "baseline directory" in line:
				actInformation["Basedirectory"] = line.split(' ')[3]
			elif "area:" in line:
				actInformation["Data-Area"] = line[1:-2].strip()
			elif "===" in line or "---" in line or line.strip() == '':
				pass	#Line is just for style reasons or empty
			elif "Executed" in line or "Elapsed time" in line or "indices in the range of" in line or "coord. info: range" in line:
				pass # Do not log timing stuff or concrete indices / coord. information
			elif "separate antennas: True" in line or "remove unused: True" in line:
				pass #as always true now, not interesting anymore

			# Detect if new Mode / Log begins or ends
			if(line[10:70] == "="*60):
				markerDetected = True
			else:
				markerDetected = False

			if "mean of output data" in line:
				observedOutputMean = True
				observedUsedRoom = False

			if "mean of used room" in line:
				observedUsedRoom = True
				observedOutputMean = False

			linecount = linecount + 1
			#print(line)

	print("%d lines of Logs analyzed" % (linecount))
	return logContent

####################################################################################################

if __name__ == "__main__":

	logContent  = getLogContent(file, directory)

	if(printTables):

		models = dict()

		for key in logContent:

			act_mode = key
			if printSections:
				print("#######################")
				print("Section: %s" % (key))
				print("#######################")
			
			logContent[key].sort(key=sortStuff, reverse=descending)

			if (act_mode == MODE.DATA.name):
				tableData = {"DataID":[], "Data-Area":[],"Coordinate Format": [], "Signal Representation": [], 
								"Room": [], "Amount of Samples": [], "Saved As":[]}

			elif(act_mode == MODE.TRAINING.name):

				if analyzeTuneTraining:
					tableData = {"DataID":[], "NetID":[], "LossID": [],"Trainingfunction":[], "Testfunction":[],
								"Data filtered afterwards":[], "Random Permutation":[], "Shuffling":[], "Normalization":[],
								"Aoa as a feature":[], "Inchannels":[], "Backprop": [], "Learningrate": [], "Momentum": [], 
								"Droprate activated":[], "Droprate":[],"Kernelsize":[],"Batchsize":[], "Epochs":[], "MeanTrainingError" : [], 
								"Amount of Trainingsamples": [], "Select from x Samples per position":[], "Hidden units":[], "Epochsize":[],"Testsize":[],"Saved As":[], "ID": []}
				else:
					tableData = {"DataID":[], "NetID":[], "LossID": [],"Trainingfunction":[], "Testfunction":[],
								"Data filtered afterwards":[], "Random Permutation":[], "Shuffling":[], "Normalization":[],
								"Aoa as a feature":[], "Inchannels":[], "Backprop": [], "Learningrate": [], "Momentum": [], 
								"Droprate activated":[], "Droprate":[],"Kernelsize":[],"Batchsize":[], "Epochs":[], "MeanTrainingError" : [], 
								"Amount of Trainingsamples": [], "Select from x Samples per position":[], "Saved As":[], "ID": []}


			elif(act_mode == MODE.TESTING.name):

				tableData = {"DataID":[], "NetID":[], "LossID": [],"Testfunction":[],"Data filtered afterwards":[],
							"Random Permutation":[], "Shuffling":[], "Normalization":[],"Aoa as a feature":[], "MeanTestingError":[], 
							"Amount of Testsamples": [], "ID": []}

			elif(act_mode == MODE.WORST.name or act_mode == MODE.BEST.name):
				
				tableData = {"DataID":[], "NetID":[], "LossID": [], "Size of Top-List":[],"Amount of Testsamples": [],"Searched Set":[], 
							"Found in columns":[], "Found in rows":[], "Error range from" :[], "To":[], "ID": []}
			
			elif(act_mode == MODE.CDF.name):
			
				tableData = {"DataID":[], "NetID":[], "LossID": [],"Data filtered afterwards":[],"Aoa as a feature":[],"Testfunction":[], 
							"First Quartile":[], "Median":[], "Third Quartile":[], "Amount of Testsamples": [],"Hidden units":[], "ID": []}
			
			elif(act_mode == MODE.BASELINE.name):
				
				tableData = {"DataID":[], "NetID":[], "LossID": [],"Testfunction":[],"Amount of Trainingsamples": [], "Amount of Testsamples": [], 
							"Error using Mean of Training Labels (Training)": [], "Error using Mean of Used Room (Testing)": [], 
							"Error using Mean of Training Labels (Testing)": [], "Error using Mean of Used Room (Training)": [], "ID": []}

			elif(act_mode == MODE.BASE_DATA_COMB.name):

				tableData = {"DataID":[], "Datadirectory":[], "Basedirectory":[], "Data was combined with sample":[],"Room":[],"Data-Area":[], 
							"Amount of Samples":[]}

			elif(act_mode == MODE.AOA_OF_DATA.name):
				tableData = {"DataID":[], "Room":[],"Amount of Samples":[],"Saved As":[]}

			else:
				tableData = {}

			# Fiter out the unwanted columns
			tableData = hideCols(tableData)

			for info in logContent[key]:

				if analyzeTuneLog:
					if info["Saved As"] in models:
						if(int(info["Epochs"]) > int(models[info["Saved As"]][0])):
							models[info["Saved As"]] = [int(info["Epochs"]), info["MeanTrainingError"]]
					else:
						models.update({info["Saved As"]:[int(info["Epochs"]), info["MeanTrainingError"]]})
						
				#print(info)

				for key in tableData:
					tableData[key].append(info[key])

				#print(tableData[key])

			if not analyzeTuneLog:
				if analyzeTuneTraining and not act_mode == MODE.CDF.name:
					pass 
				else:
					print(tabulate(tableData, headers="keys",tablefmt="grid"))


		if analyzeTuneLog:
			print("Amount of models: %d" % (len(models)))

			xlist = []

			for key in models:
				xlist.append({"Model":key, "EpochsTrained":int(models[key][0]), "TrainingsError":models[key][1],})

			xlist.sort(key=sortStuff2, reverse=descending)

			print(tabulate(xlist, headers="keys",tablefmt="plain"))
		
			
	

###################################################################################################
