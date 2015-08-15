#!/usr/bin/env python
"""
Tool for automatically generating python code from prototxt files.
"""
import caffe_pb2
from google.protobuf.text_format import Merge
import argparse

def parse_filler(param):
    s = ''
    s += 'layers.Filler('
    args = []
    if param.HasField('type'):
        args.append('type="%s"' % param.type)
    if param.HasField('value'):
        args.append('value=%s' % param.value)
    if param.HasField('min'):
        args.append('min=%s' % param.min)
    if param.HasField('max'):
        args.append('max=%s' % param.max)
    if param.HasField('mean'):
        args.append('mean=%s' % param.mean)
    if param.HasField('std'):
        args.append('std=%s' % param.std)
    if param.HasField('sparse'):
        args.append('sparse=%s' % param.sparse)
    s += ', '.join(args)
    s += ')'
    return s

def parse_kernel_stride_pad(param):
    s = ''
    args = ['']
    if param.HasField('kernel_size'):
        args.append('kernel_size=%s' % param.kernel_size)
    if param.HasField('kernel_h'):
        args.append('kernel_h=%s' % param.kernel_h)
    if param.HasField('kernel_w'):
        args.append('kernel_w=%s' % param.kernel_w)
    if param.HasField('stride'):
        args.append('stride=%s' % param.stride)
    if param.HasField('stride_h'):
        args.append('stride_h=%s' % param.stride_h)
    if param.HasField('stride_w'):
        args.append('stride_w=%s' % param.stride_w)
    if param.HasField('pad'):
        args.append('pad=%s' % param.pad)
    if param.HasField('pad_h'):
        args.append('pad_h=%s' % param.pad_h)
    if param.HasField('pad_w'):
        args.append('pad_w=%s' % param.pad_w)
        
    s += ', '.join(args)
    return s

def parse_layer(layer):
    s = ''
    s += 'layers.%s(' % layer.type
    args = []
    if layer.name:
        args.append('name="%s"' % layer.name)
    if len(layer.bottom) > 0:
        args.append('bottoms=[%s]' % ', '.join(['"%s"' % x for x in layer.bottom]))
    if len(layer.top) > 1 or len(layer.top) == 1 and layer.top[0] != layer.name:
        args.append('tops=[%s]' % ', '.join(['"%s"' % x for x in layer.top]))
    if len(layer.param) > 0:
        if layer.param[0].name:
            args.append('param_names=[%s]' % ', '.join(
                ['"%s"' % x for x in layer.param]))
        if layer.param[0].lr_mult:
            args.append('param_lr_mults=[%s]' % ', '.join(
                ['%s' % x.lr_mult for x in layer.param]))
    s += ', '.join(args)
    if len(str(layer.convolution_param)) > 0:
        s += parse_kernel_stride_pad(layer.convolution_param)
        s += ', weight_filler=%s' % parse_filler(layer.convolution_param.weight_filler)
        s += ', bias_filler=%s' % parse_filler(layer.convolution_param.bias_filler)
        s += ', num_output=%s' % layer.convolution_param.num_output
    if len(str(layer.inner_product_param)) > 0:
        s += ', weight_filler=%s' % parse_filler(layer.inner_product_param.weight_filler)
        s += ', bias_filler=%s' % parse_filler(layer.inner_product_param.bias_filler)
        s += ', num_output=%s' % layer.inner_product_param.num_output
    if len(str(layer.pooling_param)) > 0:
        s += parse_kernel_stride_pad(layer.pooling_param)
    return s
def prototxt_to_forward(prototxt):
    net = caffe_pb2.NetParameter()
    with open(prototxt, 'r') as f:
        Merge(f.read(), net)
    if net.layer:
        net_layers = net.layer
    else:
        net_layers = net.layers
    layers = []
    for layer in net_layers:
        layers.append(parse_layer(layer))
    return layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    for line in prototxt_to_forward(args.model):
        print line
main()
