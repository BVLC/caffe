#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import struct

caffe_root = sys.path[0]
caffe_root = caffe_root[:-11]
print caffe_root
sys.path.insert(0, caffe_root + 'python')

if(len(sys.argv))<=1:
    subdir = 'yolo416'
else:
    subdir = sys.argv[1]

import caffe
import numpy as np

#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_cpu()
model_filename = caffe_root+'models/yolo/'+subdir+'/yolo_deploy.prototxt'
yoloweight_filename = caffe_root+'models/yolo/'+subdir+'/yolo.weights'
caffemodel_filename = caffe_root+'models/yolo/'+subdir+'/yolo.caffemodel'
print 'model file is ', model_filename
print 'weight file is ', yoloweight_filename
print 'output caffemodel file is ', caffemodel_filename
net = caffe.Net(model_filename, caffe.TEST)
#net.forward()
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
count = 0
for layer_name, param in net.params.iteritems():
    print layer_name + '\t',
    for i in range(len(param)):
        count += np.prod(param[i].data.shape)
        #print '\nlym:'+str(np.prod(param[i].data.shape))
        print str(param[i].data.shape) + '\t',
    print
print 'count=', str(count)
params = net.params.keys()
# read weights from file and assign to the network
netWeightsInt = np.fromfile(yoloweight_filename, dtype=np.int32)
transFlag = (netWeightsInt[0] > 1000 or netWeightsInt[1] > 1000)
# transpose flag, the first 4 entries are major, minor, revision and net.seen
print 'transFlag = %r' % transFlag
netWeightsFloat = np.fromfile(yoloweight_filename, dtype=np.float32)
padflag = struct.unpack("I", netWeightsFloat[1:2].tobytes())
padoffset = 3+int(padflag[0])
netWeights = netWeightsFloat[padoffset:]
# start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
print netWeights.shape
count = 0
for pr in params:
    lidx = list(net._layer_names).index(pr)
    layer = net.layers[lidx]
    # conv_bias = None
    if count == netWeights.shape[0]:
        print "WARNING: no weights left for %s" % pr
        break
    if layer.type == 'Convolution':
        print pr + "(conv)"
        # bias
        if len(net.params[pr]) > 1:
            bias_dim = net.params[pr][1].data.shape
        else:
            bias_dim = (net.params[pr][0].data.shape[0],)
        biasSize = np.prod(bias_dim)
        conv_bias = np.reshape(netWeights[count:count + biasSize], bias_dim)
        if len(net.params[pr]) > 1:
            assert (bias_dim == net.params[pr][1].data.shape)
            net.params[pr][1].data[...] = conv_bias
            conv_bias = None
        count += biasSize
        # batch_norm
        next_layer = net.layers[lidx + 1]
        if next_layer.type == 'BatchNorm':
            bn_dims = (3, net.params[pr][0].data.shape[0])
            bnSize = np.prod(bn_dims)
            batch_norm = np.reshape(netWeights[count:count + bnSize], bn_dims)
            count += bnSize
        # weights
        dims = net.params[pr][0].data.shape
        weightSize = np.prod(dims)
        net.params[pr][0].data[...] = np.reshape(netWeights[count:count + weightSize], dims)
        count += weightSize
    elif layer.type == 'BatchNorm':
        print pr + "(batchnorm)"
        net.params[pr][0].data[...] = batch_norm[1]  # mean
        net.params[pr][1].data[...] = batch_norm[2]  # variance
        net.params[pr][2].data[...] = 1.0  # scale factor
    elif layer.type == 'Scale':
        print pr + "(scale)"
        net.params[pr][0].data[...] = batch_norm[0]  # scale
        batch_norm = None
        if len(net.params[pr]) > 1:
            net.params[pr][1].data[...] = conv_bias  # bias
            conv_bias = None
    else:
        print "WARNING: unsupported layer, " + pr
if np.prod(netWeights.shape) != count:
    print "ERROR: size mismatch: %d %d" % (count,np.prod(netWeights.shape))
else:
    print "you are right."
    net.save(caffemodel_filename)
