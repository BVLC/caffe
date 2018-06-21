#!/usr/bin/env python
#-*-coding=utf-8

import sys
sys.path.append('python')

import os

os.environ["GLOG_minloglevel"] = "3"
import caffe
import numpy as np
from prune_opts import *

# show all parameters
np.set_printoptions(threshold='nan')

if len(sys.argv) < 5:
    print "Usage: python copy_params.py <source>.prototxt <source>.caffemodel <dest>.prototxt <dest>.caffemodel"
    exit()

if len(sys.argv) < 6:
    print ("KL")
    prune_type = "kl"
else:
    if (sys.argv[5] == 'kl'):
        print ("KL")
        prune_type = "kl"
    if (sys.argv[5] == 'random'):
        print ("RANDOM")
        prune_type = "random"
    if (sys.argv[5] == 'l1'):
        print ("L1")
        prune_type = "l1"

# original caffe model
net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TRAIN)

# model to be fileed
net0 = caffe.Net(sys.argv[3],sys.argv[4],caffe.TRAIN)

#keys0 includes all layers
keys0 = net.params.keys()

#dic includes sorted index
dic = dict.fromkeys(keys0, [])

#keys1 includes layers except branch
keys1 = net.params.keys()
keys1.remove('res2a_branch1')
keys1.remove('res3a_branch1')
keys1.remove('res4a_branch1')
keys1.remove('res5a_branch1')
keys1.remove('bn3a_branch1')
keys1.remove('bn4a_branch1')
keys1.remove('bn5a_branch1')
keys1.remove('bn2a_branch1')

#keys2 includes branch layers
keys2 = ('res2a_branch1','bn2a_branch1','res3a_branch1','bn3a_branch1','res4a_branch1','bn4a_branch1','res5a_branch1','bn5a_branch1')


#drop conv weights

for i in range(0,(len(keys1)-3),2):

    layer_key_1 = keys1[i]

    # drop layer_key_1 weights
    weight_shape_layer1 = net.params[layer_key_1][0].shape

    # load weights
    if i == 0:
        weight_layer1 = net.params[layer_key_1][0].data

    else:
        weight_layer1 = weight_layer2
        weight_shape_layer1 = (weight_shape_layer1[0],net0.params[layer_key_1][0].shape[1],weight_shape_layer1[2],weight_shape_layer1[3])
        weight_layer1.shape = weight_shape_layer1

    # get every kernel kl sortedindex
    if (prune_type == 'kl'):
        weight_sum_sortedindex = prune_by_kl(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])
    elif (prune_type == 'random'):
        weight_sum_sortedindex = prune_by_random(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])
    elif (prune_type == 'l1'):
        weight_sum_sortedindex = prune_by_l1(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])

    # one kernel occupies one column
    weight_layer1.shape = (weight_shape_layer1[0],-1)

    # delete parameters accordingly
    len_need = net.params[layer_key_1][0].shape[0] - net0.params[layer_key_1][0].shape[0]
    if (len_need > 0):
        print ("[1]prune %d filters in %s"%(len_need, layer_key_1))
    weight_sum_sortedindex = weight_sum_sortedindex[0:len_need]
    weight_layer1 = np.delete(weight_layer1,weight_sum_sortedindex,axis=0)

    # save sortedindex for later use
    dic[layer_key_1] = weight_sum_sortedindex

    # write remaining weights
    weight_layer1.shape = net0.params[layer_key_1][0].shape

    for m in range(0,weight_layer1.shape[0]):
        net0.params[layer_key_1][0].data[m] = weight_layer1[m]


    # drop bn weights
    bn = keys1[i+1]
    weight_bn = net.params[bn][0].data
    weight_bn = np.delete(weight_bn,weight_sum_sortedindex,axis=0)
    for m in range(0,weight_bn.shape[0]):
        net0.params[bn][0].data[m] = weight_bn[m]


    # drop layer_key_2 weights

    # load weights
    layer_key_2 = keys1[i+2]
    weight_layer2 = net.params[layer_key_2][0].data
    weight_shape_layer2 = weight_layer2.shape
    weightshape = weight_shape_layer2[2]*weight_shape_layer2[3]
    weight_layer2.shape = (-1,weightshape)

    #delete minimum kernel
    row_delete=[]

    for n in weight_sum_sortedindex:
        for m in range(n,weight_shape_layer2[0]*weight_shape_layer2[1]+1,weight_shape_layer2[1]):
            row_delete.append(m)

    weight_layer2 = np.delete(weight_layer2,row_delete,axis=0)


#drop branch1 weights

for i in range(0,len(keys2),2):
    layer_key_0 = keys2[i]

    # load weights
    weight_layer0 = net.params[layer_key_0][0].data
    weight_shape_layer0 = net.params[layer_key_0][0].shape

    weightshape = weight_shape_layer0[2]*weight_shape_layer0[3]
    weight_layer0.shape = (-1,weightshape)

    # get sortedindex
    layer0_index = keys0.index(layer_key_0)
    layer_above = keys0[layer0_index-2]
    weight_sum_sortedindex = dic[layer_above]

    # delete minimum kernel
    row_delete=[]
    for n in weight_sum_sortedindex:
        for m in range(n,weight_shape_layer0[0]*weight_shape_layer0[1]+1,weight_shape_layer0[1]):
            row_delete.append(m)

    weight_layer0 = np.delete(weight_layer0,row_delete,axis=0)

    # get every kernel kl sortedindex
    weight_layer0.shape = (weight_shape_layer0[0],-1,weight_shape_layer0[2],weight_shape_layer0[3])
    if (prune_type == 'kl'):
        weight_sum_sortedindex = prune_by_kl(weight_layer0,weight_layer0.shape[1],weight_shape_layer0[0],weight_shape_layer0[2])
    elif (prune_type == 'random'):
        weight_sum_sortedindex = prune_by_random(weight_layer0,weight_layer0.shape[1],weight_shape_layer0[0],weight_shape_layer0[2])
    elif (prune_type == 'l1'):
        weight_sum_sortedindex = prune_by_l1(weight_layer0,weight_layer0.shape[1],weight_shape_layer0[0],weight_shape_layer0[2])

    # one kernel occupies one column
    weight_layer0.shape = (weight_shape_layer0[0],-1)

    # delete parameters accordingly
    len_need = net.params[layer_key_0][0].shape[0] - net0.params[layer_key_0][0].shape[0]
    if (len_need > 0):
        print ("[2]prune %d filters in %s"%(len_need, layer_key_0))
    weight_sum_sortedindex = weight_sum_sortedindex[0:len_need]
    weight_layer0 = np.delete(weight_layer0,weight_sum_sortedindex,axis=0)

    # write remaining weights
    weight_layer0.shape = net0.params[layer_key_0][0].shape

    for m in range(0,weight_layer0.shape[0]):
        net0.params[layer_key_0][0].data[m] = weight_layer0[m]


    # drop bn weights
    bn = keys2[i+1]
    weight_bn = net.params[bn][0].data
    weight_bn = np.delete(weight_bn,weight_sum_sortedindex,axis=0)
    for m in range(0,weight_bn.shape[0]):
        net0.params[bn][0].data[m] = weight_bn[m]

# drop res5c_branch2c weights

# drop weights related with deleted filters of above layer

# load weights
layer_key_2 = 'res5c_branch2c'
layer2_index = keys0.index(layer_key_2)
layer_above = keys0[layer2_index-2]
weight_layer2 = net.params[layer_key_2][0].data
weight_shape_layer2 = weight_layer2.shape
weightshape = weight_shape_layer2[2]*weight_shape_layer2[3]
weight_layer2.shape = (-1,weightshape)

# delete minimum kernel
row_delete=[]
weight_sum_sortedindex = dic[layer_above]
for n in weight_sum_sortedindex:
    for m in range(n,weight_shape_layer2[0]*weight_shape_layer2[1]+1,weight_shape_layer2[1]):
        row_delete.append(m)

weight_layer2 = np.delete(weight_layer2,row_delete,axis=0)


# drop res5c_branch2c weights according to its own sortedindex

layer_key_1 = layer_key_2
weight_shape_layer1 = net.params[layer_key_1][0].shape

# load weights
weight_layer1 = weight_layer2
weight_shape_layer1 = (weight_shape_layer1[0],net0.params[layer_key_1][0].shape[1],weight_shape_layer1[2],weight_shape_layer1[3])
weight_layer1.shape = weight_shape_layer1

# get every kernel kl sortedindex
if (prune_type == 'kl'):
    weight_sum_sortedindex = prune_by_kl(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])
elif (prune_type == 'random'):
    weight_sum_sortedindex = prune_by_random(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])
elif (prune_type == 'l1'):
    weight_sum_sortedindex = prune_by_l1(weight_layer1,weight_shape_layer1[1],weight_shape_layer1[0],weight_shape_layer1[2])

# one kernel occupies one column
weight_layer1.shape = (weight_shape_layer1[0],-1)

# delete 16X parameters
len_need = net.params[layer_key_1][0].shape[0] - net0.params[layer_key_1][0].shape[0]
if (len_need > 0):
    print ("[3]prune %d filters in %s"%(len_need, layer_key_1))
weight_sum_sortedindex = weight_sum_sortedindex[0:len_need]

# delete minimum kernel
weight_layer1 = np.delete(weight_layer1,weight_sum_sortedindex,axis=0)

# write remaining weights
weight_layer1.shape = net0.params[layer_key_1][0].shape

for m in range(0,weight_layer1.shape[0]):
    net0.params[layer_key_1][0].data[m] = weight_layer1[m]


# drop bn weights
bn = keys0[layer2_index+1]
weight_bn = net.params[bn][0].data
weight_bn = np.delete(weight_bn,weight_sum_sortedindex,axis=0)
for m in range(0,weight_bn.shape[0]):
    net0.params[bn][0].data[m] = weight_bn[m]


# drop fc1000 weights
layerfc = 'fc1000'
weight_layerfc = net.params[layerfc][0].data
weight_shape_layerfc = weight_layerfc.shape

# calculate the size of feature map
featuremap = weight_layerfc.shape[1] / net.params[layer_key_1][0].shape[0]
weight_layerfc.shape = (-1,featuremap)

# delete minimum kernel
row_delete=[]
for n in weight_sum_sortedindex:
    for m in range(n,weight_shape_layerfc[0]*weight_shape_layerfc[1]+1,weight_shape_layerfc[1]):
        row_delete.append(m)

weight_layerfc = np.delete(weight_layerfc,row_delete,axis=0)

# write remaining weights
weight_layerfc.shape = net0.params[layerfc][0].shape

for m in range(0,weight_layerfc.shape[0]):
    net0.params[layerfc][0].data[m] = weight_layerfc[m]

#save caffemodel
net0.save(sys.argv[4])

