__author__ = 'pittnuts'
'''
convert convolution layers in prototxt to equivalent prototxt with sparse cnn
'''
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import re
from numpy import *
import os
import caffeparser


caffe.set_device(0)
caffe.set_mode_gpu()
#prototxt_file = './models/eilab_reference_sparsenet/deploy.prototxt'
#prototxt_file = './models/eilab_reference_sparsenet/deploy.prototxt'
prototxt_file = './models/eilab_reference_sparsenet/train_val.prototxt'

#prototxt_file = 'examples/cifar10/cifar10_full.prototxt'
#prototxt_file = 'examples/cifar10/cifar10_full_train_test.prototxt'

#net = caffe.Net('./models/bvlc_reference_caffenet/deploy.prototxt','./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
#print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
#for key_idx in range(0,len(net.blobs.keys())):
#    #print net.params.keys()[key_idx]
#    key = net.blobs.keys()[key_idx]
#    if None != re.match(".*conv.*",key):
#        weights = net.params[key][0].data
#        print net.blobs.keys()[key_idx], ' -> ','kernel:',weights.shape,'blob:' ,net.blobs[net.blobs.keys()[key_idx]].data.shape

#parser = CaffeProtoPaser('../models/bvlc_reference_caffenet/deploy.prototxt')
net_parser = caffeparser.CaffeProtoParser(prototxt_file)
net_msg = net_parser.readProtoNetFile()
#print net_msg
#ilayer = net_msg.layer.add()
#ilayer.name = 'conv6'
#net_msg.layer._values.insert(1,ilayer)
#net_msg.layer._values.pop()
##del net_msg.layer[0]
#print net_msg

conv_counter = 0
'''
configuration for caffenet
'''
pads=(0,2,1,1,1)
groups=(1,2,1,2,2)
strides=(4,1,1,1,1)
kernel_sizes=(11,5,3,3,3)
m_all=(3,96,256,384,384)
'''
configuration for quick & full cifar
'''
#pads=         (2,2,2)
#groups=       (1,1,1)
#strides=      (1,1,1)
#kernel_sizes= (5,5,5)
#m_all=        (3,32,32)

layer_idx = 0
loop_layers = net_msg.layer[:] #adding : implicitly makes a copy to avoid being modified in the loop
conv_positions = []
position_idx = 0
for cur_layer in loop_layers:
    if 'Convolution'==cur_layer.type:
        print 'generating:', cur_layer.name,cur_layer.type
        conv_positions.append(position_idx)
        #p conv layer
        conv_p = net_msg.layer.add()
        conv_p.CopyFrom(cur_layer) #get a copy to make sure other parameters are consist
        conv_p.name = conv_p.name+'p'
        conv_p.bottom._values[0]=cur_layer.bottom._values[0]
        conv_p.top._values[0] = conv_p.name

        if len(conv_p.param)==0:
            lr_param = caffe_pb2.ParamSpec()
            lr_param.lr_mult = 0
            lr_param.decay_mult = 0
            conv_p.param._values.append(lr_param)
        else:
            conv_p.param._values[0].lr_mult = 0
            conv_p.param._values[0].decay_mult = 0

        try:
            #conv_p.param._values[1].lr_mult = 0
            #conv_p.param._values[1].decay_mult = 0
            del conv_p.param._values[1]
        except:
            print "Failed to operate param field: {}".format(conv_p.name)

        conv_p.convolution_param.num_output = m_all[layer_idx]
        conv_p.convolution_param.pad=pads[layer_idx]
        conv_p.convolution_param.kernel_size=1
        conv_p.convolution_param.group=groups[layer_idx]
        conv_p.convolution_param.stride=1
        conv_p.convolution_param.bias_term=False
        if conv_p.convolution_param.HasField("bias_filler"):
            conv_p.convolution_param.ClearField("bias_filler")

        #q conv layer
        conv_q = net_msg.layer.add()
        conv_q.CopyFrom(cur_layer) #get a copy to make sure other parameters are consist
        conv_q.name = conv_q.name+'q'
        conv_q.bottom._values[0]=conv_p.name
        conv_q.top._values[0] = conv_q.name
        #conv_q.param._values[0].lr_mult = 0
        #conv_q.param._values[0].decay_mult = 0
        if len(conv_q.param)==0:
            lr_param = caffe_pb2.ParamSpec()
            lr_param.lr_mult = 0
            lr_param.decay_mult = 0
            conv_q.param._values.append(lr_param)
        else:
            conv_q.param._values[0].lr_mult = 0
            conv_q.param._values[0].decay_mult = 0

        try:
            #conv_q.param._values[1].lr_mult = 0
            #conv_q.param._values[1].decay_mult = 0
            del conv_q.param._values[1]
        except:
            print "Failed to operate param field: {}".format(conv_q.name)
        conv_q.convolution_param.num_output = m_all[layer_idx]*(kernel_sizes[layer_idx]**2)
        conv_q.convolution_param.pad=0
        conv_q.convolution_param.kernel_size=kernel_sizes[layer_idx]
        conv_q.convolution_param.group=m_all[layer_idx]
        conv_q.convolution_param.stride=strides[layer_idx]
        conv_q.convolution_param.bias_term=False
        if conv_q.convolution_param.HasField("bias_filler"):
            conv_q.convolution_param.ClearField("bias_filler")

        #t conv layer
        conv_t = net_msg.layer.add()
        conv_t.CopyFrom(cur_layer) #get a copy to make sure other parameters are consist
        conv_t.name = cur_layer.top._values[0]
        conv_t.bottom._values[0]=conv_q.name
        conv_t.top._values[0] = conv_t.name
        #conv_t.param._values[0].lr_mult = 0.0
        #conv_t.param._values[0].decay_mult = 0.0
        #conv_t.param._values[1].lr_mult = 0.0
        #conv_t.param._values[1].decay_mult = 0.0
        conv_t.convolution_param.num_output = cur_layer.convolution_param.num_output#m_all[layer_idx]
        conv_t.convolution_param.pad=0
        conv_t.convolution_param.kernel_size=1
        conv_t.convolution_param.group=groups[layer_idx]
        conv_t.convolution_param.stride=1
        conv_t.convolution_param.bias_term=True

        #NEXT conv layer
        layer_idx += 1

    #next layer
    position_idx += 1

#reorder
for pos in conv_positions[::-1]:
    print pos
    l = len(net_msg.layer) #current length
    #store the p q t at the tail
    last_p = net_msg.layer._values[l-3]
    last_q = net_msg.layer._values[l-2]
    last_t = net_msg.layer._values[l-1]
    #delete original conv and insert decomposed ones
    net_msg.layer._values.pop(pos)
    net_msg.layer._values.insert(pos,last_p)
    net_msg.layer._values.insert(pos+1,last_q)
    net_msg.layer._values.insert(pos+2,last_t)
    #delete the p q t at the tail
    net_msg.layer._values.pop()
    net_msg.layer._values.pop()
    net_msg.layer._values.pop()
    pass
file_split = os.path.splitext(prototxt_file)
filepath = file_split[0]+'_scnn'+file_split[1]
file = open(filepath, "w")
if not file:
    raise IOError("ERROR (" + filepath + ")!")
file.write(str(net_msg))
file.close()

print net_msg
print conv_positions
