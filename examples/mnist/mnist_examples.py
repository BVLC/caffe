#!/usr/bin/env python

import os, struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
 
def read(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays
 
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
 
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"
 
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()
 
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
 
    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)
 
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
 
    return images, labels
 
"""
Let's read a hundred test images for figure "2" for instance

"""
CAFFE_ROOT = '../../'  # this file is expected to be in {caffe_root}/examples/mnist

images, labels = read([2], 'testing', os.path.join(CAFFE_ROOT,'data/mnist'))
 
outputs = zeros((100,28,28,1),dtype=float)
outputs[:,:,:,0] = images[0:100] * (1.0/256)
 
np.save('mnist-predict-100-twos.npy', outputs)

