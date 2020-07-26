#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides a class to visualize the results using plotting methods and libraries (e.g. input data or cdf)

"""
################################################# IMPORTS #############################################################

from colour import Color 

import matplotlib.pyplot as plt
import numpy as np
import dataManagement
import utils

########################################## Plotting methods ############################################################

"""
Plotting methods to visualize some inputs using matplotlibs
=> Manually verify if the inserted data is meaningful
"""

# --------------------------------------------- Plot Data directly -------------------------------------------------- #

# Params = 0 -> default; If != 0 => Plot as shown below
def plotData(data,title, params):
    
    if(params == 0):

        # If list: plot all of them
        if(isinstance(data,list)):

            # Use these 4 colors (in a cyclic way)
            colors = ['r','b','g','y']
            for i in range(0,len(data)):
                plt.plot(data[i],colors[i%4])
        else:
            plt.plot(data)
    else:
        plt.plot(data[0])
        plt.plot(data[1], data[2], data[3])

    plt.title(title)
    plt.show()

# --------------------------------------------- Plot Cumulative ----------------------------------------------------- #

# Plot CDF with lines for quartiles
def plotCumulative(array, title, automatedMode):

    # * 0.3 -> each square has a length of 30cm
    x = array * 0.3
    y = np.arange(len(x),dtype=float)
    y = y / len(x)

    plt.grid(color='gray', linestyle='-', linewidth=0.3)

    # Median
    median = np.median(x)
    ymarker1 = np.ones(len(x)) * 0.5     
    xmarker1 = np.arange(len(x),dtype=float) / len(x)
    xmarker1 = xmarker1 * median
    ymarker2 = np.arange(len(x),dtype=float) / len(x)
    ymarker2 = ymarker2 * 0.5
    xmarker2 = np.ones(len(x)) * median
    plt.plot(xmarker1, ymarker1,'r--')
    plt.plot(xmarker2, ymarker2,'r--')

    # First quartile
    firstquantile = np.quantile(x,0.25)
    ymarker1 = np.ones(len(x)) * 0.25     
    xmarker1 = np.arange(len(x),dtype=float) / len(x)
    xmarker1 = xmarker1 * firstquantile
    ymarker2 = np.arange(len(x),dtype=float) / len(x)
    ymarker2 = ymarker2 * 0.25
    xmarker2 = np.ones(len(x)) * firstquantile
    plt.plot(xmarker1, ymarker1,'g--')
    plt.plot(xmarker2, ymarker2,'g--')

    # Third quartile
    thirdquantile = np.quantile(x,0.75)
    ymarker1 = np.ones(len(x)) * 0.75     
    xmarker1 = np.arange(len(x),dtype=float) / len(x)
    xmarker1 = xmarker1 * thirdquantile
    ymarker2 = np.arange(len(x),dtype=float) / len(x)
    ymarker2 = ymarker2 * 0.75
    xmarker2 = np.ones(len(x)) * thirdquantile
    plt.plot(xmarker1, ymarker1,'g--')
    plt.plot(xmarker2, ymarker2,'g--')

    # Show it only, if not in automatedMode
    if not automatedMode:
        plt.step(x, y)
        plt.title(title, fontsize=20)
        plt.xlabel("Error distance (m)", fontsize=20)
        plt.ylabel("CDF", fontsize=20)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

    return [firstquantile, median, thirdquantile]

########################################################################################################################
