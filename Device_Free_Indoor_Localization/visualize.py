#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Felix Kosterhon

Provides a class to visualize the results in the given scenario/room

"""

######################################################## IMPORTS #######################################################

from PyQt5.QtWidgets import QWidget, QApplication, QPushButton
from PyQt5.QtGui import QPainter, QColor, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QTimer

import matplotlib.pyplot as plt
import numpy as np

from colour import Color 

import utils
import dataManagement

########################################## Room Visualizer #############################################################

"""
RoomVisualizer: Class to visualize the room including the position of the receiver, the true location as well as the estimated one

Example usage:
    app = QApplication(sys.argv)
    ex = Example("Office",[0,0],[1,1],[2,2],Room)
    sys.exit(app.exec_())
"""

# Visualizes the room: shows the correct as well as the estimated position
class roomVisualizer(QWidget):
    
    # Positions: Estimated = first position, real = second position
    def __init__(self, title, posReceiver, posTransmitter, positions, grid, timeout, videoMode):
        super().__init__()

        # Just for now:
        self.videoMode = videoMode

        # If a timeout is used, the window is closed after a certain amount of seconds
        if(timeout != -1):
            self.time_to_wait = timeout
            self.timer = QTimer(self)
            self.timer.setInterval(1000)    # Every 1000ms, the timer is decreased
            self.timer.timeout.connect(self.closeApplication)
            self.timer.start()
       
        # Further parameters
        self.currentSample = 0              # current sample index
        self.positions = positions          # positions contains the different samples; one sample = [posEstimated, posReal]
        self.posReceiver = posReceiver      # position of the receiver 
        self.posTransmitter = posTransmitter # position of the transmitter
        self.grid = grid                    # room / grid, which will be painted

        # General appearance parameter
        self.width = len(grid[0])
        self.height = len(grid)
        self.title = title
        
        # Initialize GUI
        self.initUI()
        
    
    # Initialize GUI
    def initUI(self):      

        # Set title
        self.setWindowTitle(self.title)

        # Set geometry depending on videoMode (=orientation) For vertical orientation only one picture is required
        if self.videoMode:
            self.setGeometry(self.width*50+50, self.width*30+150, self.height*30+50, self.height*50+150)
        else:
            self.setGeometry(self.width*50+50, self.height*50+150, self.width*50+50, self.height*50+150)

            # Two buttons to navigate through the samples:  
            self.btnNext = QPushButton("=>",self)                   # => move forward
            self.btnNext.move(self.width*25+394,self.height*50+40)
            self.btnNext.clicked.connect(self.onClickedNext)
            self.btnNext.setFont(QFont('Decorative', 25))

            self.btnPrev = QPushButton("<=",self)                   # <= move backward
            self.btnPrev.move(self.width*25+294,self.height*50+40)
            self.btnPrev.clicked.connect(self.onClickedPrev)
            self.btnPrev.setFont(QFont('Decorative', 25))

        self.show()

    # Close the application after the time ran out
    def closeApplication(self):
        self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()

    # Move forward
    def onClickedNext(self):    
        if(self.currentSample < len(self.positions) - 1):
            self.currentSample = self.currentSample + 1
        self.update()

    # Move backward
    def onClickedPrev(self):    
        if(self.currentSample > 0):
            self.currentSample = self.currentSample - 1
        self.update()

    # alternativ: Enter key for forward and backspace for backward
    def keyPressEvent(self, event): 
        key = event.key()
        
        if key in {Qt.Key_Return}:
           self.onClickedNext()
        elif key in {Qt.Key_Backspace}:
            self.onClickedPrev()


    # Will be called on self.show() and each self.update() !
    def paintEvent(self, e):       
        qp = QPainter()
        qp.begin(self)
        self.drawScenario(qp)
        qp.end()

        
    # Draw Scenario
    def drawScenario(self, qp):

        # Update sample / positions => set positions to current sample-position
        self.posReal = self.positions[self.currentSample][1]
        self.posEstimated = self.positions[self.currentSample][0]

        # Display both locations in label below the room
        text_real      = 'The true location is at'
        text_estimated = 'The predicted position is'

        # Change coordinate - referencepoint for intuitive understanding
        posRealConv = utils.changeCoordRef(self.posReal, len(self.grid))
        posEstimConv = utils.changeCoordRef(self.posEstimated, len(self.grid))

        text_real_coord = '[%2d,%2d]' % (posRealConv[0], posRealConv[1])
        text_estimated_coord = '[%2d,%2d]' % (posEstimConv[0], posEstimConv[1])

        # Print the fonts on the correct places depending on the orientation
        if self.videoMode:
            text_estimated_coord = '[%2d,%2d]' % (self.posEstimated[1]+1, self.posEstimated[0]+1)

            qp.setFont(QFont('Decorative', 25))
            qp.drawText(25, self.height*50+40,self.width*50,50,0, text_estimated )
            qp.drawText(450, self.height*50+40,self.width*50,50,0, text_estimated_coord ) 

        else:
            # Draw labels
            qp.setFont(QFont('Decorative', 25))
            qp.drawText(25, self.height*50+40,self.width*50,50,0, text_real ) 
            qp.drawText(25, self.height*50+90,self.width*50,50,0, text_estimated )
            qp.drawText(450, self.height*50+40,self.width*50,50,0, text_real_coord ) 
            qp.drawText(450, self.height*50+90,self.width*50,50,0, text_estimated_coord ) 

        # Set colors for the positions of the objects and the receiver as well as the true and estimated location
        receiver = QColor(23,25,178)    #dark blue
        transmitter = QColor(70,190,255)  #bright blue

        real= QColor(9,251,15)  # brigth green
        estimated = QColor(245, 3, 3) #red
        objects = QColor(150,150,150) #grey
        col = QColor(0, 0, 0)
        
        qp.setPen(col)

        # Draw the board depending on the orientation
        if self.videoMode:
            # Draw the board and insert the correct positions
            for i in range(0,self.height):
                for j in range(0,self.width):
                
                    if(self.grid[i][j] == -1):
                         qp.setBrush(objects)
                    else:
                        qp.setBrush(QColor(255, 255, 255))
                        
                    if([i,j] == self.posReceiver):
                        qp.setBrush(receiver)
                    elif([i,j] == self.posTransmitter):
                        qp.setBrush(transmitter)
                    elif([i,j] == self.posReal):
                        qp.setBrush(real)
                    elif([i,j] == self.posEstimated):
                        qp.setBrush(estimated)
                    
                    qp.drawRect(25+i*30, 25+(self.width -1 - j)*30, 28, 28)
        else:
            # Draw the board and insert the correct positions
            for i in range(0,self.height):
                for j in range(0,self.width):
                
                    if(self.grid[i][j] == -1):
                         qp.setBrush(objects)
                    else:
                        qp.setBrush(QColor(255, 255, 255))
                        
                    if([i,j] == self.posReceiver):
                        qp.setBrush(receiver)
                    elif([i,j] == self.posTransmitter):
                        qp.setBrush(transmitter)
                    elif([i,j] == self.posReal):
                        qp.setBrush(real)
                    elif([i,j] == self.posEstimated):
                        qp.setBrush(estimated)
                    
                    qp.drawRect(25+j*50, 25+i*50, 48, 48)

########################################## Heat Map Drawer #############################################################

# Visualizes a heatmap
# every position has a certain color depending on the average error for this position
class heatMapVisualizer(QWidget):
    
    # Positions: Estimated = first position, real = second position
    def __init__(self, title, posReceiver, posTransmitter, heatmap, grid, heatcolors, borders, debug, isError, timeout, filename):
        super().__init__()

        # If a timeout is used, the window is closed after a certain amount of seconds
        if(timeout != -1):
            self.time_to_wait = timeout
            self.timer = QTimer(self)
            self.timer.setInterval(1000) # Every 1000ms, the timer is decreased
            self.timer.timeout.connect(self.closeApplication)
            self.timer.start()

            self.filename = filename

        # Further parameters
        self.heatcolors = heatcolors
        self.heatmap = heatmap
        self.borders = borders
        self.grid = grid                    # room / grid, which will be painted

        self.isError = isError

        self.posReceiver = posReceiver      # position of the receiver 
        self.posTransmitter = posTransmitter # position of the transmitter
        self.debug = debug

        # General appearance parameter
        self.width = len(grid[0])
        self.height = len(grid)
        self.title = title


        self.borders[0] = self.heatmap[min(self.heatmap, key=self.heatmap.get)]

        # Use worst average instead of worst single estimate:
        self.borders[1] = self.heatmap[max(self.heatmap, key=self.heatmap.get)]

        # Initialize GUI
        self.initUI()
        
    def closeApplication(self):
        self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()
       
    # Initialize GUI 
    def initUI(self):      

        self.setGeometry(0, 0, self.width*50+150, self.height*50+150)
        self.setWindowTitle(self.title)
        self.showNormal()

    # Will be called on self.show() and each self.update() !
    def paintEvent(self, e):       
        qp = QPainter()
        qp.begin(self)
        self.drawScenario(qp)
        self.drawBar(qp)
        qp.end()

    # Draw a color bar, which gives the color for the certain errors
    def drawBar(self, qp):

        # If the given values represent squares, the error has to be multiplied with 30cm to convert the squares into meters
        if(self.isError):
            factor = 0.3
        else:
            factor = 1

        ratio = (len(self.grid)*50) / len(self.heatcolors)

        # Draw the colors
        for i in range(0,len(self.heatcolors)):
            c = self.heatcolors[i]

            qp.setBrush(QColor(c.get_hex()))
            pen = QPen()
            pen.setStyle(Qt.NoPen)
            qp.setPen(pen)
            qp.drawRect(self.width*50 + 50, 25+ratio*i, 48, np.ceil(ratio))

        colorRange = abs(self.borders[1] - self.borders[0])

        # Print the corresponding errors
        for i in range(0,np.ceil(len(self.grid)/2).astype(np.int64)):
            val = (colorRange * ((1+4*i)/(len(self.grid)*2))) + self.borders[0]
            text = "%.2fm" % (val*factor)
            qp.setFont(QFont('Decorative', 25))
            pen.setStyle(Qt.NoPen)
            qp.setPen(pen)
            qp.setBrush(QColor(0,0,0))
            qp.drawRect(self.width*50 + 50, 25+24+i*100, 48,1)
            col = QColor(0, 0, 0)
            qp.setPen(col)
            qp.drawText(self.width*50 + 100+10, 25+6+i*100,self.width*50,50,0, text) 


    # Draw Scenario
    def drawScenario(self, qp):

        # Set colors for the positions of the objects and the receiver as well as the true and estimated location
        receiver = QColor(23,25,178)    #dark blue
        transmitter = QColor(70,190,255)  #bright blue

        real= QColor(9,251,15)  # brigth green
        estimated = QColor(245, 3, 3) #red
        objects = QColor(150,150,150) #grey
        col = QColor(0, 0, 0)
        
        qp.setPen(col)

        dataProcessor = dataManagement.DataProcessing(False)

        colorRange = abs(self.borders[1] - self.borders[0])

        if(self.debug):
            print(colorRange)
            print(len(self.heatcolors))
            print(len(self.heatmap))
        
        if(self.isError):
            factor = 0.3
        else:
            factor = 1

        # Print the best, middle and worst errors
        textBest = "%.2fm" % (self.borders[0]*factor)
        textMiddle = "%.2fm" % ((self.borders[0] + (colorRange/2))*factor)
        textWorst = "%.2fm" % (self.borders[1]*factor)

        # Print legend
        qp.setFont(QFont('Decorative', 25))
        qp.drawText(25, self.height*50+50,self.width*50,50,0, "Errors:") 
        qp.drawText(240, self.height*50+50,self.width*50,50,0, textBest )
        qp.drawText(440, self.height*50+50,self.width*50,50,0, textMiddle) 
        qp.drawText(640, self.height*50+50,self.width*50,50,0, textWorst) 

        qp.drawText(770, self.height*50+50,self.width*50,50,0, "|")         

        qp.drawText(825, self.height*50+50,self.width*50,50,0, "Positions:") 
        qp.drawText(1090, self.height*50+50,self.width*50,50,0, "RX") 
        qp.drawText(1240, self.height*50+50,self.width*50,50,0, "TX") 
        
        c = self.heatcolors[0]
        qp.setBrush(QColor(c.get_hex()))
        qp.drawRect(25+150, self.height*50+50-6, 48, 48)

        c = self.heatcolors[len(self.heatcolors)//2]
        qp.setBrush(QColor(c.get_hex()))
        qp.drawRect(25+350, self.height*50+50-6, 48, 48)

        c = self.heatcolors[len(self.heatcolors)-1]
        qp.setBrush(QColor(c.get_hex()))
        qp.drawRect(25+550, self.height*50+50-6, 48, 48)

        qp.setBrush(receiver)
        qp.drawRect(25+1000, self.height*50+50-6, 48, 48)

        qp.setBrush(transmitter)
        qp.drawRect(25+1150, self.height*50+50-6, 48, 48)

        # Draw the board and insert the correct positions
        for i in range(0,self.height):
            for j in range(0,self.width):
            
                if(self.grid[i][j] == -1):
                     qp.setBrush(objects)
                elif(self.grid[i][j] == 0):
                    qp.setBrush(QColor(255, 255, 255))
                elif([i,j] == self.posReceiver):
                    qp.setBrush(receiver)
                elif([i,j] == self.posTransmitter):
                    qp.setBrush(transmitter)
                else:
                    qp.setBrush(QColor(255, 255, 255))

                qp.drawRect(25+j*50, 25+i*50, 48, 48)
                
        for key in self.heatmap:

            pos = dataProcessor.findPos(key, self.grid)
            val = self.heatmap[key]

            if(self.debug):
                print(pos)

            val = (val - self.borders[0]) / colorRange
            if(self.debug):
                print(val)

            index = round(val.item() * 500)
            if(self.debug):
                print(index-1)

            c = self.heatcolors[index-1]

            qp.setBrush(QColor(c.get_hex()))

            qp.drawRect(25+pos[1]*50, 25+pos[0]*50, 48, 48)

        print("Indices from %.2fm over %.2fm to %.2fm" % (self.borders[0]*factor, (self.borders[0] + (colorRange/2))*factor, self.borders[1]*factor))

########################################## Hit Map Drawer ##############################################################

class hitMapVisualizer(QWidget):
    
    # Positions: Estimated = first position, real = second position
    def __init__(self, title, posReceiver, posTransmitter, hitmap, grid, heatcolors, debug, timeout, filename):
        super().__init__()

        # Store the inputs (each scenario one)
        self.inputList = []

        for key in hitmap:
            self.inputList.append(key)

        self.actIndex = 0

        # If a timeout is used, the window is closed after a certain amount of seconds
        if(timeout != -1):
            self.time_to_wait = timeout
            self.timer = QTimer(self)
            self.timer.setInterval(1000)
            self.timer.timeout.connect(self.closeApplication)
            self.timer.start()

            self.filename = filename

        # Further parameters
        self.heatcolors = heatcolors
        self.hitmap = hitmap
        self.grid = grid                    # room / grid, which will be painted

        self.posReceiver = posReceiver      # position of the receiver 
        self.posTransmitter = posTransmitter # position of the transmitter
        self.debug = debug

        # General appearance parameter
        self.width = len(grid[0])
        self.height = len(grid)
        self.title = title

        minVal = 1000
        maxVal = 0

        # Calcuate Min and Max for each of them
        for key in hitmap:
            actMin = np.min(hitmap[key][0])
            actMax = np.max(hitmap[key][0])

            if actMin < minVal:
                minVal = actMin

            if actMax > maxVal:
                maxVal = actMax

        self.borders = [minVal, maxVal]

        print(self.borders)
        
        # Init GUI
        self.initUI()
       
    # Close the application 
    def closeApplication(self):
        self.time_to_wait -= 1
        if self.time_to_wait <= 0:
            self.close()
        
    # Move forward
    def onClickedNext(self):    
        if(self.actIndex < len(self.inputList) - 1):
            self.actIndex = self.actIndex + 1
        self.update()

    # Move backward
    def onClickedPrev(self):   
        if(self.actIndex > 0):
            self.actIndex = self.actIndex - 1
        self.update()

    # alternativ: Enter key for forward and backspace for backward
    def keyPressEvent(self, event): 
        key = event.key()
        
        if key in {Qt.Key_Return}:
           self.onClickedNext()
        elif key in {Qt.Key_Backspace}:
            self.onClickedPrev()

    # init GUI
    def initUI(self):      

        self.setGeometry(0, 0, self.width*50+150+80, self.height*50+150+25)
        self.setWindowTitle(self.title)
        

        # Two buttons to navigate through the samples:  
        self.btnNext = QPushButton("=>",self)                   # => move forward
        self.btnNext.move(self.width*25+644,self.height*50+118)
        self.btnNext.clicked.connect(self.onClickedNext)
        self.btnNext.setFont(QFont('Decorative', 25))

        self.btnPrev = QPushButton("<=",self)                   # <= move backward
        self.btnPrev.move(self.width*25+544,self.height*50+118)
        self.btnPrev.clicked.connect(self.onClickedPrev)
        self.btnPrev.setFont(QFont('Decorative', 25))

        self.showNormal()

    # Will be called on self.show() and each self.update() !
    def paintEvent(self, e):       
        qp = QPainter()
        qp.begin(self)
        self.drawScenario(qp)
        self.drawBar(qp)
        qp.end()

    # Draw the color bar and the corresponding labels
    def drawBar(self, qp):

        ratio = (len(self.grid)*50) / len(self.heatcolors)

        for i in range(0,len(self.heatcolors)):
            c = self.heatcolors[i]

            qp.setBrush(QColor(c.get_hex()))
            pen = QPen()
            pen.setStyle(Qt.NoPen)
            qp.setPen(pen)
            qp.drawRect(self.width*50 + 50, 25+ratio*i, 48, np.ceil(ratio))

        colorRange = abs(self.borders[1] - self.borders[0])

        # Adapt color range automatically
        modVal = 1
        while(int(colorRange) / modVal > 8):
            modVal = modVal + 1

        for i in range(0, int(colorRange)+1):

            if((i % modVal) == (int(colorRange) % modVal)):

                roundedval = i

                indexForVal = int(((roundedval - self.borders[0]) / colorRange) * 500)
                
                posY = int((indexForVal * 800) / 500)

                text = "%d" % (roundedval)
                qp.setFont(QFont('Decorative', 25))
                pen.setStyle(Qt.NoPen)
                qp.setPen(pen)
                qp.setBrush(QColor(0,0,0))
                qp.drawRect(self.width*50 + 50, 25+24+posY, 48,1)
                col = QColor(0, 0, 0)
                qp.setPen(col)
                qp.drawText(self.width*50 + 100+10, 25+6+posY,self.width*50,50,0, text) 

    # Draw Scenario
    def drawScenario(self, qp):

        # Set colors for the positions of the objects and the receiver as well as the true and estimated location
        receiver = QColor(23,25,178)    #dark blue
        transmitter = QColor(70,190,255)  #bright blue

        real= QColor(9,251,15)  # brigth green
        estimated = QColor(245, 3, 3) #red
        objects = QColor(150,150,150) #grey
        col = QColor(0, 0, 0)
        qp.setPen(col)
        dataProcessor = dataManagement.DataProcessing(False)

        # get actual hits 
        key = self.inputList[self.actIndex]

        outlierNum = self.hitmap[key][2]
        totalNum = self.hitmap[key][1] + self.hitmap[key][2]

        actMin = np.min(self.hitmap[key][0])
        actMax = np.max(self.hitmap[key][0])

        # If uncommented: Set manually
        # actMax = 120

        self.borders = [actMin, actMax]

        colorRange = abs(self.borders[1] - self.borders[0])

        textBest = "%.2f" % (self.borders[0])
        textMiddle = "%.2f" % ((self.borders[0] + (colorRange/2)))
        textWorst = "%.2f" % (self.borders[1])
        
        # Print legend
        qp.setFont(QFont('Decorative', 25))

        qp.drawText(25, self.height*50+50,self.width*50,50,0, "Hits per Position from") 
        qp.drawText(500, self.height*50+50,self.width*50,50,0, textBest)
        qp.drawText(620, self.height*50+50,self.width*50,50,0, "to") 
        qp.drawText(800, self.height*50+50,self.width*50,50,0, textWorst) 

        qp.drawText(920, self.height*50+50,self.width*50,50,0, "|")         

        qp.drawText(970, self.height*50+50,self.width*50,50,0, "Positions:") 
        qp.drawText(970+265, self.height*50+50,self.width*50,50,0, "RX") #+265
        qp.drawText(970+265+150, self.height*50+50,self.width*50,50,0, "TX") #+150
        
        
        c = self.heatcolors[0]
        qp.setBrush(QColor(c.get_hex()))
        qp.drawRect(25+400, self.height*50+50-6, 48, 48)

        c = self.heatcolors[len(self.heatcolors)-1]
        qp.setBrush(QColor(c.get_hex()))
        qp.drawRect(25+700, self.height*50+50-6, 48, 48)
        
        qp.setBrush(receiver)
        qp.drawRect(25+1150, self.height*50+50-6, 48, 48)

        qp.setBrush(transmitter)
        qp.drawRect(25+1300, self.height*50+50-6, 48, 48)

        textInfo = "%d Points outside the visible area (of %d in total)" % (outlierNum,totalNum)
        qp.drawText(25, self.height*50+50+50+25,self.width*50,50,0, textInfo )

        # Draw the board and insert the correct positions
        for i in range(0,self.height):
            for j in range(0,self.width):
            
                if(self.grid[i][j] == -1):
                     qp.setBrush(objects)
                elif(self.grid[i][j] == 0):
                    qp.setBrush(QColor(255, 255, 255))
                elif([i,j] == self.posReceiver):
                    qp.setBrush(receiver)
                elif([i,j] == self.posTransmitter):
                    qp.setBrush(transmitter)
                elif([i,j] == dataProcessor.findPos(key, self.grid)):
                    qp.setBrush(real)

                else:
                    qp.setBrush(QColor(255, 255, 255))

                qp.drawRect(25+j*50, 25+i*50, 48, 48)
        
        # Print colors according to the amount of hits for this position
        hitsForKey = self.hitmap[key][0]
        for x in range(0,len(hitsForKey)):
            for y in range(0,len(hitsForKey[0])):

                pos = [x,y]

                val = hitsForKey[x][y]

                val = (val - self.borders[0]) / colorRange

                index = round(val.item() * 500)
                if index == 500: index = 499

                c = self.heatcolors[index]
                
                if(pos == self.posReceiver or pos == self.posTransmitter):
                    col = QColor(255, 255, 255)
                    qp.setPen(col)
                    count = "%d" % (hitsForKey[x][y])
                    qp.drawText(25+pos[1]*50, 28+pos[0]*50, 48, 48, 20, count)
                    col = QColor(0, 0, 0)
                    qp.setPen(col)
                elif(pos == dataProcessor.findPos(key, self.grid)):
                    count = "%d" % (hitsForKey[x][y])
                    qp.drawText(25+pos[1]*50, 28+pos[0]*50, 48, 48, 20, count)
                else:
                    qp.setBrush(QColor(c.get_hex()))
                    qp.drawRect(25+pos[1]*50, 25+pos[0]*50, 48, 48)