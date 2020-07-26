#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Contains all enumerators: Modes, Office columns / row, loss functions, coordinates, signal representations, backpropagation, iqconverter, data

"""

from enum import Enum

# Different Modes for the program
class MODE(Enum):
    DATALOADER = 0          # Load data
    BASELOADER = 1          # Load Baseline data from given directory
    COAXLOADER = 2          # Load measurements of coax cable measurements
    APPLYMASK = 3           # Apply the specified mask and store the data
    APPLYBASETODATA = 4     # Apply prev. loaded Baseline to Data and save it combined
    GETAOAOFDATA = 5        # Calculate the MUSIC angles of the given data
    TRAINING = 6            # Train network
    TESTING = 7             # Calculate the mean error by inserting testsamples
    EXTREMESEARCH = 8       # Search for best/worst estimate
    ANALYZETUNEMODELS = 9   # Analyze the given TUNE models
    VISUALIZATION = 10      # Visualize some random/ extreme samples
    CDFPLOT = 11            # Create CDF for a certain network - data
    CREATEHEATMAP = 12      # Heat map of average error
    CREATEHITMAP = 13       # Create a hitmap
    REFINEDATALOADER = 14   # Load the data for the Refinement
    REFINENETWORK = 15      # Refine the network by only training the fully connected layer
    INSTANT = 16            # Instant Mode: "Live" estimation of the incoming data via a pretrained network
    USER_DEFINED = 17       # Call user defined stuff

# Column areas => split it horizontally
# Room divided into columns -> determined in roomParameters.py      VeryLeft | Left | Center | Right | VeryRight
# LeftArea = VeryLeft + Left, RightArea = VeryRight + Right, BigCenter = Left + Center + Right
class ROOM_COL(Enum):
    COMPLETE = 0
    EMPTY = 1
    VERYLEFT = 2
    LEFT = 3
    CENTER = 4
    RIGHT = 5
    VERYRIGHT = 6
    LEFTAREA = 7
    RIGHTAREA = 8
    BIGCENTER = 9

# Row areas => split it vertically
# Room divided into rows -> determined in roomParameters.py      Top | Upper | Mid | Lower | Bottom
# UpperCompl = Top + Upper, LowerCompl = Bottom + Lower, MidCompl = Lower + Mid + Upper
class ROOM_ROW(Enum):
    COMPLETE = 0
    EMPTY = 1
    TOP = 2
    UPPER = 3
    MID = 4
    LOWER = 5
    BOTTOM = 6
    UPPERCOMPL = 7
    LOWERCOMPL = 8
    MIDCOMPL = 9

# Different Lossfunctions
class LOSSFUNCTIONS(Enum):
    ANGLEDIST = 0           # MSE (Mean-Square-Error) of angle and distance difference
    CARTESIAN = 1           # MSE of cartesian coordinates -> Distance (sqrt of MSE)
    MSELOSS = 2             # Use standard function MSELOSS
    CARTESIAN_DIST = 3      # Cartesian distance

# Coordinate formats
class COORDINATES(Enum):
    CARTESIAN = 0           # Cartesian coordinate
    POLAR = 1               # Polar coordinates

# signal representations / input representations
class SIGNALREPRESENTATIONS(Enum):
    IQ = 0                  # iq samples
    AMPL_PHASE = 1          # amplitude & phase
    AMPL_PHASE_DIFF = 2     # amplitude & phase
    AMPL_PHASE_UNWRAP = 3   # amplitude & phase unwrapped

# Backprop algorithms
class BACKPROP(Enum):
    SGD = 0                 # Stochastic Gradient Descent
    ADAM = 1                # Adam

# Converter Version (Used Kernel and CSI extractor)
class IQCONVERTER(Enum):
    OLD = 0                 # Old workflow
    NEW = 1                 # New Nexmon CSI extractor (https://github.com/seemoo-lab/nexmon_csi)

# What part of the data is requested (used by data loader)
class DATA(Enum):
    ALL = 0                 # Get all of the data
    TRAINING = 1            # Get the data used for training / which will be used for training
    TESTING = 2             # Get all the data except the data used for training => Test set (validation and evaluation)