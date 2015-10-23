__author__ = 'pittnuts'
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from PIL import Image
import caffe
import sys
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
from os import system
from caffe_apps import *
import argparse
import bottleneck as bn
import os
import re
from  caffeparser import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--filepattern', type=str, required=True)
    args = parser.parse_args()
    dir=args.dir
    filepattern=args.filepattern

    for f in os.listdir(dir):
        if os.path.isfile(f) and re.match(filepattern,f):
            print "processing {}".format(f)
            protoparser = CaffeProtoParser (f)
            blob = protoparser.readBlobProto()
            blob_reshape = array(blob.data).reshape((blob.shape.dim[1],blob.shape.dim[2]*blob.shape.dim[3]))#reshape((blob.shape[1],blob.shape[2]*blob.shape[3]))
            thre = 1
            #blob_reshape = array(blob.data).reshape((blob.shape.dim[0],blob.shape.dim[1]*blob.shape.dim[2]*blob.shape.dim[3]))
            #thre = 0.0001
            binary_blob=abs(blob_reshape)>thre
            plt.subplot(1,2,1)
            #plt.imshow(blob_reshape,cmap=plt.gray())
            plt.imshow(blob_reshape,cmap="Greys")
            zero_out(blob_reshape,thre)
            print np.sum(blob_reshape,axis=1)==0
            plt.xlabel('gray map')
            plt.title("{}\n{:.2f}".format(os.path.splitext(f)[0],get_sparsity(blob_reshape,thre)))
            plt.subplot(1,2,2)
            #plt.imshow(binary_blob,cmap=plt.gray())
            plt.imshow(binary_blob,cmap="Greys")
            plt.xlabel('binary map')
            plt.show()




