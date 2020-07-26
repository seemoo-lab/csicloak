#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides various methods to work on the data.
It processes the measurements as well as the labels and combines them to the input data.


"""

# ------------------------------------------- IMPORT ----------------------------------------------------------- #

from scipy.signal import find_peaks

import os
import sys
import math
import numpy as np
import h5py
import torch
import random

import enumerators
import essentialMethods
import plottingUtils
import utils

##################################################################################################################
############################################ CLASS CSI DATASET ###################################################
##################################################################################################################

class CSIDataset:
    
    # normalization can be done here
    # If params is a list -> mean, min and max already known and should be applied here!
    # This is necessary e.g. to normalize the test set in the same way as the training set -> no information leakage!
    def __init__(self, data, normalize, params):
        self.data = data
        self.mean = 0
        self.minval = 0
        self.maxval = 0
        
        helpdata = []
        if(normalize):
            if(not isinstance(params, list)):
                for i in range(0,len(data)):
                    helpdata.append(data[i][0].numpy())
                helpdata = np.array(helpdata)
                mean = helpdata.mean(axis=0)

                self.mean = torch.from_numpy(mean)
                self.maxval = np.max(helpdata)
                self.minval = np.min(helpdata)
            else:
                self.mean = params[0]
                self.minval = params[1]
                self.maxval = params[2]

            for i in range(0,len(data)):
                self.data[i][0] = (data[i][0] - self.minval) / (self.maxval - self.minval)

        
    # implement these methods to be used as set
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    # get properties of the set
    def getMean(self):
        return self.mean

    def getMax(self):
        return self.maxval

    def getMin(self):
        return self.minval

#####################################################################################################################
########################################### CLASS DATAPROCESSING ####################################################
#####################################################################################################################

class DataProcessing:

    # init DataProcessing unit
    def __init__(self, debug):
        self.debug = debug
        
    # Find given value in matrix / room and return the position
    # It the value is not in the matrix -> return [-1,-1] (invalid indices)
    # [Y Coord, X Coord] ! (Row, Column)
    # => Values refer to origin top left
    def findPos(self,val, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == val:
                    return [i,j]
        return [-1, -1]

    # Inverse operation of previous method
    # Given the position, find the value at this position in the matrix/room
    def getNumberOfPos(self, pos, room):
        return room[pos[0].astype(int)][pos[1].astype(int)]

    # Convert cartesian coordinates to polar coordinates
    def convertToPolarCoord(self,pos, orig):
        # r = radius and a = angle
        r = math.sqrt((orig[0]-pos[0])**2 + (pos[1]-orig[1])**2)
        a = (360 + math.degrees(math.atan2(orig[0]-pos[0],pos[1]-orig[1]))) % 360
        return [r,a]
    
    # Search for the position of a range of values in the given matrix
    # Return the list of positions in the specified type of coordinates
    def getCoordOfRange(self,start,end,ref,matrix, coordtype,posExceptions):

        res = np.zeros(((end-start+1-len(posExceptions)),2))
        ref_coord = self.findPos(ref,matrix)
        counter = 0
        for i in range(start,end+1):
            if i not in posExceptions:
                tmp_coord = self.findPos(i,matrix)
                if(coordtype == enumerators.COORDINATES.POLAR):
                    tmp_coord = self.getPolarCoord(tmp_coord, ref_coord)
                res[counter] = tmp_coord # Count from 0 !
                counter = counter+1
        return res
    
    # Gets the amount of samples for all positions according to the given mask
    def createListSamplesPerLocation(self, coordInfo, room, mask):
        targets = self.getCoordOfRange(coordInfo[0],coordInfo[1],coordInfo[2],room, enumerators.COORDINATES.CARTESIAN, coordInfo[4])
        samplesList = []

        for i in range(0,len(targets)):
            coord = targets[i]
            samplesList.append(mask[int(coord[0])][int(coord[1])])

        return samplesList


    """
    This loads all HDF5 Files of a certain directory. Therefore, the convertFiles script has to be run previously
    to change the .mat files to .hdf5
    The content is extracted and returned in the file structure
    """
    def loadHDF5FilesOfDirectory(self,directory, converter):
        files = []
        filenames = []

        # collect all filenames which end with .hdf5
        for filename in os.listdir(directory):
            if filename.endswith(".hdf5"):
                filenames.append(os.path.join(directory,filename))
                if(self.debug):   
                    print(os.path.join(directory,filename))
                    print(len(files))
                    h5py.File(os.path.join(directory,filename),'r')
        
        # Sort it
        filenames.sort()
        
        # Open each file and extract the content
        for filename in filenames:
            f_tmp = h5py.File(filename,'r')
            keys=[]
            f_tmp.visit(keys.append)
            x = np.array(f_tmp[keys[2]][()])

            print("Read File: %s" % (filename))

            if(self.debug):
                print("Filename: %s, keys: %s" % (filename, str(keys)))

            if(converter == enumerators.IQCONVERTER.OLD):
                files.append(np.reshape(x, (-1,2)))
            elif(converter == enumerators.IQCONVERTER.NEW):
                files.append(np.swapaxes(x, 0, 2))

            f_tmp.close()

        return files
    
    
    # Combine the sorted input (from hdf5 files) and sorted targets (from position map / matrix / root) in one dict
    # One target (from the map) should be mapped to multiple files (filesPerTarget parameter) as each position was captured multiple times
    # Do NOT consider the indices found in exception => Use only parts of the room / matrix
    # If the signal shouldnot be represented as IQ sample (as given in mat files), the signal will be converted here to amplitude-phase-notation
    def createDictList(self, files, targets, filesPerTarget, signalrepr, exception, room, converter):
        
        data = []
        thresh_unwrapping = 1.5*np.pi

        C = np.zeros([255, 256])
        for i in range(0,len(C)):
            for j in range(0,len(C[0])):
                if j == i:
                    C[i][j] = -1
                elif i == j-1:
                    C[i][j] = 1

        # Sanity check: length of files has to correspond to the filesPerTarget and the amount of files
        if len(files) != (len(targets) * filesPerTarget):
            return -1

        # Iterate through all targets
        for i in range(0,len(targets)):
            target = targets[i]

            # check if the target should be included or not
            if(self.getNumberOfPos(target,room) not in exception):
                if(self.debug):
                    print("Current valid index: " + str(self.getNumberOfPos(target,room)))

                # Each target is used for multiple positions
                for j in range(0,filesPerTarget):
                    file = files[i*filesPerTarget+j]

                    # If amplitude-phase notation -> change the representation here
                    if(signalrepr == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE or signalrepr == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE_DIFF or signalrepr == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE_UNWRAP): 

                        if(converter == enumerators.IQCONVERTER.NEW):
                            phases = np.zeros([len(file), len(file[0]), len(file[0][0])])

                            for core in range(0,len(file)):
                                for rx in range(0, len(file[0])):

                                    for k in range(0,len(file[0][0])):
                                        inphase = file[core][rx][k][0]
                                        quadr = file[core][rx][k][1]
                                        ampl = np.sqrt(inphase**2 + quadr**2)
                                        phase = math.atan2(quadr,inphase)
                                        file[core][rx][k][0] = ampl
                                        file[core][rx][k][1] = phase
                                        phases[core][rx][k] = phase

                                    tester1 = file[core][rx]
                                    tester = np.zeros([2,len(tester1)])                    

                        # old workflow
                        elif(converter == enumerators.IQCONVERTER.OLD):
                            # Change each data point of the sample
                            phases = np.zeros(len(file))

                            for k in range (0,len(file)):
                                inphase = file[k][0]
                                quadr = file[k][1]
                                ampl = np.sqrt(inphase**2 + quadr**2)
                                phase = math.atan2(quadr,inphase)
                                file[k][0] = ampl
                                file[k][1] = phase
                                phases[k] = phase

                            if(self.debug):
                                essentialMethods.plotData(phases,"Phases",0)

                        # Unwrap the phase
                        if(signalrepr == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE_UNWRAP):

                            # Depending on the used iq-converter, the input format is slightly different
                            if(converter == enumerators.IQCONVERTER.NEW):
                                for core in range(0,len(file)):
                                    phases_act = phases[core]
                                    deltas = [C.dot(phases_act[0]), C.dot(phases_act[1]), C.dot(phases_act[2]), C.dot(phases_act[3])]
                                    counter = [0,0,0,0]

                                    for d in range(0,len(deltas[0])):
                                        for k in range(0,len(deltas)):
                                            if(deltas[k][d] > thresh_unwrapping):
                                                counter[k] = counter[k] + 1
                                            if(deltas[k][d] < -1*thresh_unwrapping):
                                                counter[k] = counter[k] - 1
                                            phases_act[k][d+1] = phases_act[k][d+1] - 2*np.pi*counter[k]

                                    for rx in range(0, len(file[0])):
                                        for k in range(0,len(file[0][0][0])):
                                            file[core][rx][k][1] = phases_act[rx][k]

                            elif(converter == enumerators.IQCONVERTER.OLD):
                                phases = [phases[0:256], phases[256:512], phases[512:768], phases[768:1024]]
                                deltas = [C.dot(phases[0]), C.dot(phases[1]), C.dot(phases[2]), C.dot(phases[3])]
                                counter = [0,0,0,0]

                                for d in range(0,len(deltas[0])):
                                    for k in range(0,len(deltas)):
                                        if(deltas[k][d] > thresh_unwrapping):
                                            counter[k] = counter[k] + 1
                                        if(deltas[k][d] < -1*thresh_unwrapping):
                                            counter[k] = counter[k] - 1
                                        phases[k][d+1] = phases[k][d+1] - 2*np.pi*counter[k]

                                phases = np.concatenate([phases[0],phases[1],phases[2],phases[3]],0)

                                if(self.debug):
                                    essentialMethods.plotData(phases,"Phases unwrapped",0)

                                for k in range (0,len(file)):
                                    file[k][1] = phases[k]

                        if(self.debug): 
                            print("Ampl: ",ampl," und Phase: ",phase)

                    data.append({"data": file, "target": target})

        return data
    
    # filters the dictionary and applies a sanity check
    def filterDict(self, dictionary, samplesList, filesPerTarget, errorLog):

        print(len(dictionary))

        if(len(dictionary) % filesPerTarget != 0):
            utils.printFailure("Error happened in filterDict @ dataManagement: Dictionary length is not a multiple of FilesPerTarget")
            errorLog.append("Error happened in filterDict @ dataManagement: Dictionary length is not a multiple of FilesPerTarget")

        if(len(dictionary) != (len(samplesList) * filesPerTarget)):
            utils.printFailure("Error happened in filterDict @ dataManagement: Dictionary length is not correct")
            errorLog.append("Error happened in filterDict @ dataManagement: Dictionary length is not correct")

        data = []
        for i in range(0,len(samplesList)):
            for j in range(0,samplesList[i]):
                data.append(dictionary[i*filesPerTarget+j])

        print("Length before: %d (dict length); afterwards: %d (data len)" % (len(dictionary), len(data)))
        return data, errorLog

    # reshapes the matrix into TX x RX x SC x 2
    def reshapeMatrix(self, data, mode):

        data_new = []

        for i in range(0,len(data)):
            data_item = []
            target_item = []

            for tx in range(0,len(data[0]["data"])):
                for rx in range(0, len(data[0]["data"][0])):

                    # core in file = RX; rxss = spatial strean => TX
                    tmp_1 = data[i]["data"][rx][tx]

                    tmp = np.zeros([2,len(tmp_1)])
                    
                    for k in range(0,len(tmp_1)):
                        tmp[0][k] = tmp_1[k][0]
                        tmp[1][k] = tmp_1[k][1]
                    
                    data_item.append(tmp[0])
                    data_item.append(tmp[1])

                if(mode == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE_DIFF):

                    idx_offset = core*8
                    # only shows first TX!
                    if(self.debug):
                        essentialMethods.plotData(np.concatenate((data_item[idx_offset+0], data_item[idx_offset+2], data_item[idx_offset+4], data_item[idx_offset+6]),0), "Even indices",0)
                        essentialMethods.plotData(np.concatenate((data_item[idx_offset+1], data_item[idx_offset+3], data_item[idx_offset+5], data_item[idx_offset+7]),0), "Odd indices",0)
                        essentialMethods.plotData(data_item[idx_offset+1], "index 1",0)
                        essentialMethods.plotData(np.concatenate((np.sqrt(data_item[idx_offset+0]**2 + data_item[idx_offset+1]**2), np.sqrt(data_item[idx_offset+2]**2 + data_item[idx_offset+3]**2), 
                                np.sqrt(data_item[idx_offset+4]**2 + data_item[idx_offset+5]**2), np.sqrt(data_item[idx_offset+6]**2 + data_item[idx_offset+7]**2)),0), "Abs distance with even/odd indices",0)
                        essentialMethods.plotData(data_item[idx_offset+1] - data_item[idx_offset+3], "Phase difference: Antenna 1 - Antenna 2",0)
                        essentialMethods.plotData(data_item[idx_offset+5] - data_item[idx_offset+7], "Phase difference: Antenna 2 - Antenna 4",0)

                    # Use Phase differences instead of phase directly
                    tmp1 = data_item[idx_offset+1]
                    tmp3 = data_item[idx_offset+3]
                    tmp5 = data_item[idx_offset+5]
                    tmp7 = data_item[idx_offset+7]

                    data_item[idx_offset+1] = tmp1 - tmp3
                    data_item[idx_offset+3] = tmp3 - tmp7
                    data_item[idx_offset+5] = tmp1 - tmp5
                    data_item[idx_offset+7] = tmp1 - tmp7

                    # Test if correct data is being inserted
                    if(self.debug):                
                        essentialMethods.plotData(data_item[idx_offset+1], "Phase difference: Antenna 1 - Antenna 2",0)
                        essentialMethods.plotData(data_item[idx_offset+3], "Phase difference: Antenna 2 - Antenna 4",0)

            # Add target and data to the data array
            target_item = data[i]["target"]
            data_new.append({"data": data_item, "target": target_item})
        return data_new

    # Method to separate the antennas
    # Instead of 2 channels of 1024 -> 8 channels of 256 (each antenna has 2 channels, one for real and one for complex values)
    # Only necessary for the old workflow
    def separateAntennas(self, data, mode):
        
        data_separated = []
        separator = len(data[0]["data"])//4

        # For each sample in the data
        for i in range(0,len(data)):
            data_item = []
            target_item = []

            # Have a look at each antenna individually
            for k in range(0,4):
                tmp = data[i]["data"]
                tmp = torch.tensor(tmp).transpose(0,1)
                
                # add 2 channels for each antenna 
                data_item.append(tmp[0][separator*k:separator*(k+1)].numpy())
                data_item.append(tmp[1][separator*k:separator*(k+1)].numpy())

            if(mode == enumerators.SIGNALREPRESENTATIONS.AMPL_PHASE_DIFF):

                if(self.debug):
                    essentialMethods.plotData(np.concatenate((data_item[0], data_item[2], data_item[4], data_item[6]),0), "Even indices",0)
                    essentialMethods.plotData(np.concatenate((data_item[1], data_item[3], data_item[5], data_item[7]),0), "Odd indices",0)
                    essentialMethods.plotData(data_item[1], "index 1",0)
                    essentialMethods.plotData(np.concatenate((np.sqrt(data_item[0]**2 + data_item[1]**2), np.sqrt(data_item[2]**2 + data_item[3]**2), 
                            np.sqrt(data_item[4]**2 + data_item[5]**2), np.sqrt(data_item[6]**2 + data_item[7]**2)),0), "Abs distance with even/odd indices",0)
                    essentialMethods.plotData(data_item[1] - data_item[3], "Phase difference: Antenna 1 - Antenna 2",0)
                    essentialMethods.plotData(data_item[5] - data_item[7], "Phase difference: Antenna 2 - Antenna 4",0)

                # Use Phase differences instead of phase directly
                tmp1 = data_item[1]
                tmp3 = data_item[3]
                tmp5 = data_item[5]
                tmp7 = data_item[7]

                data_item[1] = tmp1 - tmp3
                data_item[3] = tmp3 - tmp7
                data_item[5] = tmp1 - tmp5
                data_item[7] = tmp1 - tmp7

                # Test if correct data is being inserted
                if(self.debug):
                    essentialMethods.plotData(np.concatenate((data_item[0], data_item[2], data_item[4], data_item[6]),0), "Even indices",0)
                    essentialMethods.plotData(np.concatenate((data_item[1], data_item[3], data_item[5], data_item[7], data_item[8], data_item[9]),0), "Odd indices",0)
                
                    essentialMethods.plotData(data_item[1], "Phase difference: Antenna 1 - Antenna 2",0)
                    essentialMethods.plotData(data_item[3], "Phase difference: Antenna 2 - Antenna 4",0)

            # Add target and data to the data array
            target_item = data[i]["target"]
            data_separated.append({"data": data_item, "target": target_item})
        return data_separated
        
    # convert the target as well as data of each entry to a tensor and put both tensors of each entry in a list
    # Includes shuffling => Not anymore; now handled afterwards
    def convertArrayToTensor(self, array, converter):
        data = []
        for i in range(0,len(array)):
            input = torch.tensor(array[i]["data"])
            label = torch.tensor(array[i]["target"])
            data.append([input,label])
        return data

    # Change the upper and lower sideband!
    # Only necessary for old workflow
    def swapsidebands(self, data):

        # For each entry
        for i in range(0,len(data)):

            # for each channel (2 if antennas not separated, else: 8)
            for j in range(0,len(data[i][0])):

                tmp = torch.zeros(256)
                tmp[:128] = data[i][0][j][128:] # -> swap the halves of the data entries
                tmp[128:] = data[i][0][j][:128]
                data[i][0][j] = tmp
        return data

    # Identify the pilots by searching for peaks in the amplitude
    def getPilotIndices(self, data, coordtype, samplesToAnalyze, converter, errorLog):

        peakset = {}
        threshold = 0.3

        if(len(data) < samplesToAnalyze):
            samplesToAnalyze = len(data)

        # Iterate over some samples
        for s in range(0, samplesToAnalyze):
            sample = data[s][0]

            # Analyze peaks depending on type of data (IQ or Amplitude/Phase)
            for i in range(0, len(sample)//2):
                sample = np.array(sample)
                absDistance = np.zeros([len(sample[0])])

                # IQ samples -> amplitude has to be calculated; Ampl / Phase => Ampl direct accessible
                if(coordtype == enumerators.SIGNALREPRESENTATIONS.IQ):
                    for j in range(0,len(sample[0])):
                        absDistance[j] = np.sqrt(sample[2*i][j]**2 + sample[2*i+1][j]**2) 
                else:
                    absDistance = sample[2*i]

                # Find peaks in amplitude to identify pilots
                if(converter == enumerators.IQCONVERTER.OLD):
                    peakindices, _ = find_peaks(absDistance,threshold=threshold)
                elif(converter == enumerators.IQCONVERTER.NEW):
                    peakindicesPos, _ = find_peaks(absDistance,threshold=100)
                    peakindicesNeg, _ = find_peaks(-absDistance,threshold=100)
                    
                    peakindices = np.concatenate([np.asarray(peakindicesPos), np.asarray(peakindicesNeg)])
                #self.debug = True
                if(self.debug):
                    print("Peak indices: ", peakindices)
                    plottingUtils.plotData([absDistance, peakindices, absDistance[peakindices], "x"],"Found peaks", 1)
                    plottingUtils.plotData([absDistance, peakindicesNeg, absDistance[peakindicesNeg], "x"],"Found neg. peaks", 1)

                # convert all found indices
                for k in range(0,len(peakindices)):
                    index = utils.indicesToSC(peakindices[k])
                    
                    if(self.debug):
                        print("Verify Conversion: ", peakindices[k], " => ", index, " => ", utils.scToIndices(index))
                
                    # Count amount of peaks identified over all samples; if found -> one occurence more
                    if index in peakset:
                        peakset[index] = peakset[index] + 1
                    else:
                        peakset[index] = 1

        #DC -> no pilots possible here
        peakset[0] = 0
        peakset[-1] = 0
        peakset[1] = 0

        # Only consider the 8 most frequently found indices -> very likely correspond to the 8 pilots
        peakset = sorted(peakset.items(), key=lambda item: item[1], reverse=True)
        peakset = peakset[0:8]

        # Only the indices are now important, not the amount of found occurences
        peakset = list(zip(*peakset))   # zip(*array) => unzip array
        peakset = sorted(peakset[0])    #sort them

        if(self.debug):
            print("Peakset: ", peakset)

        # Sanity check
        if(len(peakset) != 8):
            utils.printFailure("Error happened in getPilotIndices @ dataManagement: Amount of peaks not correct")
            errorLog.append("Error happened in getPilotIndices @ dataManagement: Amount of peaks not correct")

        # CHECK SYMMETRY -> 2nd sanity check
        abspeakset = np.unique(np.abs(np.array(peakset)))
        if len(abspeakset) != 4:
            utils.printFailure("Error happened in getPilotIndices @ dataManagement: Peaks not symmetric")
            print(abspeakset)
            print(peakset)
            errorLog.append("Error happened in getPilotIndices @ dataManagement: Peaks not symmetric")

        return abspeakset, errorLog

    # Remove the pilots and unused channels
    # pilots are removed based on estimated offset (should be zero)
    def removePilotsAndUnusedChannelsFast(self, data, pilots, offset, area, rmPilots):
        d = np.concatenate([-1*np.array(area), area],0)
        d.sort()
        d = np.array(d) + 128 # Map in range 0-255

        before = np.sqrt(data[0][0][0]**2 + data[0][0][1]**2)

        # If pilots are not removed -> only remove the unused channel and use the full range +/- 2 until +/- 122
        if(rmPilots):
            pilots = np.concatenate([-1*np.array(pilots), pilots],0)
            pilots.sort()
            pilots = np.array(pilots) + 128

        # Skips exactly the subcarrier, which should be eliminated
        for i in range(0,len(data)):
            if(rmPilots):
                data[i][0] = (torch.cat((data[i][0][:,d[0]:pilots[0]],data[i][0][:,(pilots[0]+1):pilots[1]],data[i][0][:,(pilots[1]+1):pilots[2]],data[i][0][:,(pilots[2]+1):pilots[3]], 
                    data[i][0][:,(pilots[3]+1):d[1]+1],data[i][0][:,(d[2]):pilots[4]],data[i][0][:,(pilots[4]+1):pilots[5]],data[i][0][:,(pilots[5]+1):pilots[6]],
                    data[i][0][:,(pilots[6]+1):pilots[7]],data[i][0][:,(pilots[7]+1):(d[3]+1)]),dim=1)).to(torch.float32)
            else:
                data[i][0] = (torch.cat((data[i][0][:,d[0]:(d[1]+1)],data[i][0][:,d[2]:(d[3]+1)]),dim=1)).to(torch.float32)         

        if(self.debug):
            essentialMethods.plotData([np.sqrt(data[0][0][0]**2 + data[0][0][1]**2),before],["Before","After"],0)

        return data

#####################################################################################################################
###################################### APPLY MASKS ##################################################################
#####################################################################################################################

# Apply a mask on the data and filter the positions according to it
def applyMaskOnData(data, coordInfo, filesPerTarget, room, mask, debug, errorLog):
    dataProcessor = DataProcessing(debug)

    sampleList = dataProcessor.createListSamplesPerLocation(coordInfo, room, mask)
    data, errorLog = dataProcessor.filterDict(data, sampleList, filesPerTarget, errorLog)
    return data, errorLog

# Filters the data and removes the data, which is in the excluded Area
def filterData(data, room, excludedArea):
    filtered_data = []

    print(excludedArea)

    for entry in data:

        if(room[int(entry[1][0].item())][int(entry[1][1].item())] not in excludedArea):
            filtered_data.append(entry)

    return filtered_data

#####################################################################################################################
########################################### DATA PIPELINE ###########################################################
#####################################################################################################################

# Directory = /Path/To/dataDirectory
# coordInfo = [Start, Stop, Receiver, Transmitter] => [1, 128, -7] bzw, [1, 98, -7]
# Room = Room
# coordMode: 0 (Cartesian); 1 => ENUM COORDINATES
# FilesperTarget = 50 bzw 40
# Signalrepresentations => ENUM SIGNALREPRESENTATIONS (IQ or Ampl/Phase)
# Exception: Constraints for the room
# Pilots: pilot subcarrier
# Datarange: subcarrier, which carry data e.g. [2,122]
# Shuffling: -> Now handled after Dataloading 
# RmPilots: remove pilot subcarrier from data
# RmUnused: remove all unused subcarrier (which do not carry any meaningful information => Datarange)

def getDataFromRawPipeline(directory, coordInfo, room, coordMode, filesPerTarget,
    signalrepr, exception, pilots, datarange, debug, 
    rmPilots, rmUnused, samplesToAnalyze, converter, inchannels, errorLog):
    
    dataProcessor = DataProcessing(debug)

    # Params: stores the lengths of all data directories
    params = []

    if(isinstance(directory, list)):    #if True -> multiple directories -> load data and combine it

        dictionary = []

        for i in range(0,len(directory)):
            # Identify the positions of the targets in range and load them
            targets = dataProcessor.getCoordOfRange(coordInfo[i][0],coordInfo[i][1],coordInfo[i][2],room, coordMode, coordInfo[i][4])

            # Load the .hdf5 files
            files = dataProcessor.loadHDF5FilesOfDirectory(directory[i], converter)

            # Combine both into a dictionary
            dictObj = dataProcessor.createDictList(files,targets,filesPerTarget[i],signalrepr, exception, room, converter)
            
            # If necessary, split the data for the areas of the 4 antennas -> 2 channels for each antenna (one for compl + one for real values)
            if(converter == enumerators.IQCONVERTER.OLD):
                dictObj = dataProcessor.separateAntennas(dictObj, signalrepr)

            # Here => add new converter method and RX/TX filtering
            if(converter == enumerators.IQCONVERTER.NEW):
                dictObj = dataProcessor.reshapeMatrix(dictObj, signalrepr)

            dictionary = np.concatenate([dictionary, dictObj],0)
            params.append(len(dictObj))

    else:

        # Identify the positions of the targets in range and load them
        targets = dataProcessor.getCoordOfRange(coordInfo[0],coordInfo[1],coordInfo[2],room, coordMode, coordInfo[4])

        # Load the .hdf5 files
        files = dataProcessor.loadHDF5FilesOfDirectory(directory, converter)

        print("Files loaded")

        print(len(targets))
        print(len(files))

        # Combine both into a dictionary
        dictionary = dataProcessor.createDictList(files,targets,filesPerTarget,signalrepr, exception, room, converter)

        print("Dictionary created")

        del targets
        del files

        # Here => add new converter method and RX/TX filtering
        if(converter == enumerators.IQCONVERTER.NEW):
            dictionary = dataProcessor.reshapeMatrix(dictionary, signalrepr)

        # If necessary, split the data for the areas of the 4 antennas -> 2 channels for each antenna (one for compl + one for real values)
        if(converter == enumerators.IQCONVERTER.OLD):
            dictionary = dataProcessor.separateAntennas(dictionary, signalrepr)

        params = len(dictionary)

    # Convert Array to a tensor and shuffle it if desired
    data = dataProcessor.convertArrayToTensor(dictionary, converter)

    del dictionary
    print("Converted to Tensor ")

    # For the old workflow, the sidebands have to be switched
    if(converter == enumerators.IQCONVERTER.OLD):
        data = dataProcessor.swapsidebands(data)
    
    # Identify pilots by searching for peaks in amplitude
    foundPilots, errorLog = dataProcessor.getPilotIndices(data, signalrepr, samplesToAnalyze, converter, errorLog)
    
    # Find offset by subtracting found from expected pilots -> afterwards, only one number should be left as
    # distances between the pilots subcarrier always match
    offset = np.unique(foundPilots - pilots)
    
    if(len(offset) > 1):
        utils.printFailure("Error happened in dataPipeline @ dataManagement: Offset not identiable")
        errorLog.append("Error happened in dataPipeline @ dataManagement: Offset not identiable")

    # Offset is the first and only entry
    offset = offset[0]
    
    #offset = 0
    print("Pilot offset identified: %d" % (offset))

    if(rmUnused):
        # Remove the pilots and unused channels 
        # => Remove the noise of the data
        data = dataProcessor.removePilotsAndUnusedChannelsFast(data, pilots, offset, datarange, rmPilots)

    del dataProcessor

    return [data, params, errorLog]


####################################################################################################
########################### INSTANT DATA PROCESSING W/O TARGETS ####################################
####################################################################################################

# Provides a complete pipeline for the instant mode
# Similar to the general workflow
# This includes error handling, if the given file cannot be found -> transmission error
def getInstantData(directory, filename, signalrepr, exception, pilots, datarange, debug, 
    rmPilots, rmUnused, room, converter, packetsToCombine):

    files = []
    dataProcessor = DataProcessing(debug)

    # All inputs are processed
    for i in range(1,packetsToCombine+1):

        filename = filename.split("__")[0] + "__" + str(i) + ".hdf5"
        print(filename)

        # try to open the file
        try:
            f_tmp = h5py.File(os.path.join(directory,filename),'r')
        except OSError:
            print("Catched!")
            # Stop it and return the error
            return -1,-1
        keys=[]
        f_tmp.visit(keys.append)
        x = np.array(f_tmp[keys[2]][()])

        # Get the data in the correct format
        if(converter == enumerators.IQCONVERTER.OLD):
            files.append(np.reshape(x, (-1,2)))
        elif(converter == enumerators.IQCONVERTER.NEW):
            files.append(np.swapaxes(x, 0, 2))

    targets = np.array([[0,0]])

    # Combine both into a dictionary
    dictionary = dataProcessor.createDictList(files,targets,packetsToCombine,signalrepr, exception, room, converter)

    # Here => add new converter method and RX/TX filtering
    if(converter == enumerators.IQCONVERTER.NEW):
        dictionary = dataProcessor.reshapeMatrix(dictionary, signalrepr)

    # If necessary, split the data for the areas of the 4 antennas -> 2 channels for each antenna (one for compl + one for real values)
    if(converter == enumerators.IQCONVERTER.OLD):
        dictionary = dataProcessor.separateAntennas(dictionary, signalrepr)

    params = len(dictionary)

    # Convert Array to a tensor (and shuffle it if desired)
    data = dataProcessor.convertArrayToTensor(dictionary, converter)

    # For the old workflow, the sidebands have to be switched
    if(converter == enumerators.IQCONVERTER.OLD):
        data = dataProcessor.swapsidebands(data)

    # Assume offset 0
    offset = 0

    if(rmUnused):
        # Remove the pilots and unused channels 
        data = dataProcessor.removePilotsAndUnusedChannelsFast(data, pilots, offset, datarange, rmPilots)

    f_tmp.close()

    del dataProcessor

    # Returns the data as well as the length
    return [data, params]
