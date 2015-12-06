# -*- coding: utf-8 -*-
"""
Training with data from memory, inspired by http://nbviewer.ipython.org/github/BVLC/caffe/blob/tutorial/examples/01-learning-lenet.ipynb
Useful because of siamese setup: avoids unnecessary disk access.

Created on Wed Sep  2 17:22:21 2015

@author: ch
"""

import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import caffe
from NetDataset import NetDataset
import sys


# plot training loss and test results
def plotStats(trainLoss, testAcc, figure=None):

    nLoss = trainLoss.shape[0]
    nAcc = testAcc.shape[0]    
    
    if (figure is None):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  
    else:        
        fig,ax1,ax2 = figure
    
    if (nAcc > 1):
        factor = round((nLoss-1)/(nAcc-1))
        
        ax1.plot(np.arange(nLoss), trainLoss)
        ax2.plot(factor * np.arange(nAcc), testAcc, 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')
        
    return [fig,ax1,ax2]
                   

### task: train siamese network in python
# parameters
dataFolder = './db'
solverDef = 'lrfacenet-solver-32.prototxt'

gpu = False 
imgDim = 32  # width and height of image

nTrainIter = 500
testInterval = 100
nTestBatches = 10


# make settings
dataFolderTrain = dataFolder + '/train'
dataFolderTest = dataFolder + '/test'
if gpu:
    caffe.set_mode_gpu()
    print('GPU mode')
else:
    caffe.set_mode_cpu()
    print('CPU mode')


### step 1: load data
print('{:s} - Load train data'.format(str(datetime.datetime.now()).split('.')[0]))
trainDataset = NetDataset()
trainDataset.targetSize = imgDim
trainDataset.loadImageData(dataFolderTrain)
trainDataset.printStatus = False

print('{:s} - Load test data'.format(str(datetime.datetime.now()).split('.')[0]))
testDataset = NetDataset()
testDataset.targetSize = imgDim
testDataset.flipAugmentation = False
testDataset.shiftAugmentation = False
testDataset.loadImageData(dataFolderTest)
testDataset.printStatus = False


### step 2: prepare net
print('{:s} - Creating net'.format(str(datetime.datetime.now()).split('.')[0]))
solver = caffe.SGDSolver(solverDef)
# print net structure
# each output is (batch size, feature dim, spatial dim)
print('                      Net data/blob sizes:')
for k, v in solver.net.blobs.items():
    print '                       ', k, v.data.shape
print('                      Net weight sizes:')
for k, v in solver.net.params.items():
    print '                       ', k, v[0].data.shape


### step 3: train net with dynamically created data
print('{:s} - Training'.format(str(datetime.datetime.now()).split('.')[0]))

# loss and accuracy will be stored in the log
trainLoss = np.zeros(nTrainIter+1)
testAcc = np.zeros(int(np.ceil(nTrainIter / testInterval))+1)

# the main solver loop
testNetData, testNetLabels = testDataset.getNextVerficationBatch()
solver.test_nets[0].set_input_arrays(testNetData, testNetLabels)
for trainIter in range(nTrainIter+1):
    netData, netLabels = trainDataset.getNextVerficationBatch()
    solver.net.set_input_arrays(netData, netLabels)
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    trainLoss[trainIter] = solver.net.blobs['loss'].data
    
    # abort if training diverged
    if (math.isnan(trainLoss[trainIter])):
       break
        
    # run a full test every so often
    # Caffe can also do this for us and write to a log, but we show here
    # how to do it directly in Python, where more complicated things are easier.
    if (trainIter % testInterval == 0):
        print('{:s} - Iteration {:d} - Loss {:.4f} - Testing'.format(str(datetime.datetime.now()).split('.')[0], trainIter, trainLoss[trainIter]))
        distancesAnchorPositives = []
        distancesAnchorNegatives = []
                
        # testing with memory interface
        # collect test results
        testDataset.classPointer = 0
        testDataset.loops = 0
        for i in range(nTestBatches):
            testNetData, testNetLabels = testDataset.getNextVerficationBatch()
            solver.test_nets[0].set_input_arrays(testNetData, testNetLabels)
            solver.test_nets[0].forward()
        
            # accuracy test for verification
            ft1 = solver.test_nets[0].blobs['feat'].data
            ft2 = solver.test_nets[0].blobs['feat_p'].data
            ft3 = solver.test_nets[0].blobs['feat_pp'].data
            
            curDist = np.sum((ft1 - ft2)**2, axis=1) # euclidean distance between anchor and positive
            distancesAnchorPositives = np.concatenate((distancesAnchorPositives, curDist))    
            curDist = np.sum((ft1 - ft3)**2, axis=1) # euclidean distance between anchor and negative
            distancesAnchorNegatives = np.concatenate((distancesAnchorNegatives, curDist))    
    
        # search for best threshold and use that accuracy
        bestAccuracy = -float('inf')
        for i in range(len(distancesAnchorPositives)):
            curThresh = distancesAnchorPositives[i]
            prediction = distancesAnchorPositives <= curThresh
            accuracy = np.mean(prediction == distancesAnchorPositives)
            if (accuracy > bestAccuracy):
                bestAccuracy = accuracy
    
        testAcc[trainIter // testInterval] = bestAccuracy
        print('                      accuracy: {:.3f}'.format(bestAccuracy))        
        
    # end testing
# end train loop

## plot final stats
plotStats(trainLoss, testAcc)    

## save plot stats
np.save('lrfacenet_{:s}_loss.npy'.format(str(datetime.datetime.now()).split('.')[0]).replace(':','-'), trainLoss)
np.save('lrfacenet_{:s}_accuracy.npy'.format(str(datetime.datetime.now()).split('.')[0]).replace(':','-'), testAcc)
            
# done
print('{:s} - Done'.format(str(datetime.datetime.now()).split('.')[0]))