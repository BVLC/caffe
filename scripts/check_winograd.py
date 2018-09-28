import os
import re
import math
import sys
import numpy as np

pycaffe = os.path.split(os.path.realpath(__file__))[0] + '/../python'
sys.path.insert(0, pycaffe)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

def check(model):
    caffe.set_mode_cpu()
    test_net = caffe.Net(model, caffe.TEST)
    layers = list(test_net._layer_names)
    conv_layers = []
    for idx in range(0, len(layers)):
        if test_net.layers[idx].type == "Convolution":
            conv_layers.append(layers[idx])

    top_map = test_net.top_names
    bottom_map = test_net.bottom_names
    channel_map = {}
    for k, v in top_map.items(): # layer name : top blobs
        if k in conv_layers:
            top_blob = v[0]
            bottom_blob = bottom_map[k][0]
            channel_map[k] = (test_net.blobs[top_blob].shape[1], test_net.blobs[bottom_blob].shape[1])

    base_net = caffe_pb2.NetParameter()
    with open(model) as f:
        s = f.read()
        txtf.Merge(s, base_net)

    winograd_convolutions = []
    for index in range(0, len(base_net.layer)):
        l = base_net.layer[index]
        if l.type == "Convolution":
            kernel_h = 0
            kernel_w = 0
            if len(l.convolution_param.kernel_size) != 0:
                kernel_h = kernel_w = l.convolution_param.kernel_size[0]
                if len(l.convolution_param.kernel_size) > 1:
                    kernel_h = l.convolution_param.kernel_size[0]
                    kernel_w = l.convolution_param.kernel_size[1]
            else:
                if l.convolution_param.HasField("kernel_h"):
                    kernel_h = l.convolution_param.kernel_h
                if l.convolution_param.HasField("kernel_w"):
                    kernel_w = l.convolution_param.kernel_w

            group = l.convolution_param.group
            stride_h = 1
            stride_w = 1
            if len(l.convolution_param.stride) != 0:
                if len(l.convolution_param.stride) == 1:
                    stride_h = stride_w = l.convolution_param.stride[0]
                else:
                    stride_h = l.convolution_param.stride[0]
                    stride_w = l.convolution_param.stride[1]
            else:
                if l.convolution_param.HasField("stride_h"):
                    stride_h = l.convolution_param.stride_h
                if l.convolution_param.HasField("stride_w"):
                    stride_w = l.convolution_param.stride_w

            dilate_h = 1
            dilate_w = 1
            if len(l.convolution_param.dilation) != 0:
                if len(l.convolution_param.dilation) == 1:
                    dilate_h = dilate_w = l.convolution_param.dilation[0]
                else:
                    dilate_h = l.convolution_param.dilation[0]
                    dilate_w = l.convolution_param.dilation[1]
 
            oc = channel_map[l.name][0]
            ic = channel_map[l.name][1]

            pad_h = 0
            pad_w = 0
            if len(l.convolution_param.pad) != 0:
                if len(l.convolution_param.pad) == 1:
                    pad_h = pad_w = l.convolution_param.pad[0]
                else:
                    pad_h = l.convolution_param.pad[0]
                    pad_w = l.convolution_param.pad[1]
            else:
                if l.convolution_param.HasField("pad_h"):
                    pad_h = l.convolution_param.pad_h
                if l.convolution_param.HasField("pad_w"):
                    pad_w = l.convolution_param.pad_w
            
            if kernel_h == 3 and kernel_w  == 3 \
               and group == 1 \
               and oc % 16 == 0 and ic % 16 == 0 \
               and stride_h == 1 and stride_w == 1 \
               and dilate_h == 0 and dilate_w == 0 \
               and pad_h < 2 and pad_w < 2:
                   winograd_convolutions.append(l.name)

    return winograd_convolutions
                   
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usgae: python check_winograd.py $prototxt"
        sys.exit(0)
    print check(sys.argv[1])
