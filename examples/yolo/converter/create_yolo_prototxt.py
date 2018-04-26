# -*- coding: utf-8 -*-
from ConfigParser import ConfigParser
from collections import OrderedDict
import argparse
import logging
import os
import sys

class CaffeLayerGenerator(object):
    def __init__(self, name, ltype):
        self.name = name
        self.bottom = []
        self.top = []
        self.type = ltype
    def get_template(self):
        return """
layer {{{{
  name: "{}"
  type: "{}"
  bottom: "{}"
  top: "{}"{{}}
}}}}""".format(self.name, self.type, self.bottom[0], self.top[0])

class CaffeInputLayer(CaffeLayerGenerator):
    def __init__(self, name, channels, width, height):
        super(CaffeInputLayer, self).__init__(name, 'Input')
        self.channels = channels
        self.width = width
        self.height = height
    def write(self, f):
        f.write("""
input: "{}"
input_shape {{
  dim: 1
  dim: {}
  dim: {}
  dim: {}
}}""".format(self.name, self.channels, self.width, self.height))

class CaffeConvolutionLayer(CaffeLayerGenerator):
    def __init__(self, name, filters, ksize=None, stride=None, pad=None, bias=True):
        super(CaffeConvolutionLayer, self).__init__(name, 'Convolution')
        self.filters = filters
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.bias = bias
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if not self.bias: opts.append('bias_term: false')
        param_str = """
  convolution_param {{
    num_output: {}{}
  }}""".format(self.filters, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))

class CaffePoolingLayer(CaffeLayerGenerator):
    def __init__(self, name, pooltype, ksize=None, stride=None, pad=None, global_pooling=None):
        super(CaffePoolingLayer, self).__init__(name, 'Pooling')
        self.pooltype = pooltype
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.global_pooling = global_pooling
    def write(self, f):
        opts = ['']
        if self.ksize is not None: opts.append('kernel_size: {}'.format(self.ksize))
        if self.stride is not None: opts.append('stride: {}'.format(self.stride))
        if self.pad is not None: opts.append('pad: {}'.format(self.pad))
        if self.global_pooling is not None: opts.append('global_pooling: {}'.format('True' if self.global_pooling else 'False'))
        param_str = """
  pooling_param {{
    pool: {}{}
  }}""".format(self.pooltype, '\n    '.join(opts))
        f.write(self.get_template().format(param_str))

class CaffeInnerProductLayer(CaffeLayerGenerator):
    def __init__(self, name, num_output):
        super(CaffeInnerProductLayer, self).__init__(name, 'InnerProduct')
        self.num_output = num_output
    def write(self, f):
        param_str = """
  inner_product_param {{
    num_output: {}
  }}""".format(self.num_output)
        f.write(self.get_template().format(param_str))

class CaffeBatchNormLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeBatchNormLayer, self).__init__(name, 'BatchNorm')
    def write(self, f):
        param_str = """
  batch_norm_param {
    use_global_stats: true
  }"""
        f.write(self.get_template().format(param_str))

class CaffeScaleLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeScaleLayer, self).__init__(name, 'Scale')
    def write(self, f):
        param_str = """
  scale_param {
    bias_term: true
  }"""
        f.write(self.get_template().format(param_str))

class CaffeReluLayer(CaffeLayerGenerator):
    def __init__(self, name, negslope=None):
        super(CaffeReluLayer, self).__init__(name, 'ReLU')
        self.negslope = negslope
    def write(self, f):
        param_str = ""
        if self.negslope is not None:
            param_str = """
  relu_param {{
    negative_slope: {}
  }}""".format(self.negslope)
        f.write(self.get_template().format(param_str))

class CaffeDropoutLayer(CaffeLayerGenerator):
    def __init__(self, name, prob):
        super(CaffeDropoutLayer, self).__init__(name, 'Dropout')
        self.prob = prob
    def write(self, f):
        param_str = """
  dropout_param {{
    dropout_ratio: {}
  }}""".format(self.prob)
        f.write(self.get_template().format(param_str))

class CaffePowerLayer(CaffeLayerGenerator):
    def __init__(self,name,scale):
        super(CaffePowerLayer, self).__init__(name, 'Power')
        self.scale = scale
    def write(self, f):
        param_str="""
  power_param {{
    scale: {}
  }}""".format(self.scale)
        f.write(self.get_template().format(param_str))

class CaffeEltWiseLayer(CaffeLayerGenerator):
    def __init__(self,name,operation): # need 2 blob to do operation
        super(CaffeEltWiseLayer, self).__init__(name, 'Eltwise')
        self.operation = operation
    def write(self, f):
        param_str="""
  bottom: "{}"
  eltwise_param{{
    operation: {}
  }}""".format(self.bottom[1],self.operation)
        f.write(self.get_template().format(param_str))

class CaffeSoftmaxLayer(CaffeLayerGenerator):
    def __init__(self, name):
        super(CaffeSoftmaxLayer, self).__init__(name, 'Softmax')
    def write(self, f):
        f.write(self.get_template().format(""))

class CaffeProtoGenerator:
    def __init__(self, name):
        self.name = name
        self.sections = []
        self.lnum = 0
        self.layer = None
    def add_layer(self, l):
        self.sections.append( l )
    def add_input_layer(self, items):
        self.lnum = 0
        lname = "data"
        self.layer = CaffeInputLayer(lname, items['channels'], items['width'], items['height'])
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def update_last_convolution_layer(self):
        self.sections[len(self.sections)-1].pad = 0
    def add_convolution_layer(self, items):
        self.lnum += 1
        prev_blob = self.layer.top[0]
        lname = "conv"+str(self.lnum)
        filters = items['filters']
        ksize = items['size'] if 'size' in items else None
        stride = items['stride'] if 'stride' in items else None
        pad = items['pad'] if 'pad' in items else None
        bias = not bool(items['batch_normalize']) if 'batch_normalize' in items else True
        self.layer = CaffeConvolutionLayer( lname, filters, ksize=ksize, stride=stride, pad=pad, bias=bias )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_innerproduct_layer(self, items):
        self.lnum += 1
        prev_blob = self.layer.top[0]
        lname = "fc"+str(self.lnum)
        num_output = items['output']
        self.layer = CaffeInnerProductLayer( lname, num_output )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_pooling_layer(self, ltype, items, global_pooling=None):
        prev_blob = self.layer.top[0]
        lname = "pool"+str(self.lnum)
        ksize = items['size'] if 'size' in items else None
        stride = items['stride'] if 'stride' in items else None
        pad = items['pad'] if 'pad' in items else None
        self.layer = CaffePoolingLayer( lname, ltype, ksize=ksize, stride=stride, pad=pad, global_pooling=global_pooling )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_batchnorm_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "bn"+str(self.lnum)
        self.layer = CaffeBatchNormLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_scale_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "scale"+str(self.lnum)
        self.layer = CaffeScaleLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_leaky_relu_layer(self,items):
        prev_blob = self.layer.top[0]
        lname = "relu"+str(self.lnum)
        self.layer = CaffeReluLayer( lname , 0.1)
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( prev_blob)
        self.add_layer( self.layer )
    def add_relu_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "relu"+str(self.lnum)
        self.layer = CaffeReluLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( prev_blob )     # loopback
        self.add_layer( self.layer )
    def add_relu_separate_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "relu"+str(self.lnum)
        self.layer = CaffeReluLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname ) # not done in place so create new blob
        self.add_layer( self.layer )
    def add_power_layer(self,items,scale=0.08):
        prev_blob = self.layer.bottom[0] # not the top since it use same previous blob
        prev_relu = self.layer.top[0] # save the same level relu
        lname = "power"+str(self.lnum)
        self.layer = CaffePowerLayer( lname, scale)
        self.layer.bottom.append( prev_blob )
        self.layer.bottom.append (prev_relu ) # save the same level relu, not gonna created since create_layer only use index 0
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_eltwise_layer(self,items,operation='SUM'):
        prev_blob = self.layer.top[0]
        prev_relu = self.layer.bottom[1] # get the relu from power layer
        lname = "eltwise"+str(self.lnum)
        self.layer = CaffeEltWiseLayer( lname, operation)
        self.layer.bottom.append( prev_blob )
        self.layer.bottom.append( prev_relu )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def add_dropout_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "drop"+str(self.lnum)
        self.layer = CaffeDropoutLayer( lname, items['probability'] )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( prev_blob )     # loopback
        self.add_layer( self.layer )
    def add_softmax_layer(self, items):
        prev_blob = self.layer.top[0]
        lname = "prob"
        self.layer = CaffeSoftmaxLayer( lname )
        self.layer.bottom.append( prev_blob )
        self.layer.top.append( lname )
        self.add_layer( self.layer )
    def finalize(self, name):
        self.layer.top[0] = name    # replace
    def write(self, fname):
        with open(fname, 'w') as f:
            f.write('name: "{}"'.format(self.name))
            for sec in self.sections:
                sec.write(f)
        logging.info('{} is generated'.format(fname))

###################################################################33
class uniqdict(OrderedDict):
    _unique = 0
    def __setitem__(self, key, val):
        if isinstance(val, OrderedDict):
            self._unique += 1
            key += "_"+str(self._unique)
        OrderedDict.__setitem__(self, key, val)

def convert(cfgfile, ptxtfile,approximate):
    #
    parser = ConfigParser(dict_type=uniqdict)
    parser.read(cfgfile)
    netname = os.path.basename(cfgfile).split('.')[0]
    #print netname
    gen = CaffeProtoGenerator(netname)
    for section in parser.sections():
        _section = section.split('_')[0]
        if _section in ["crop", "cost"]:
            continue
        #
        batchnorm_followed = False
        relu_followed = False
        items = dict(parser.items(section))
        if 'batch_normalize' in items and items['batch_normalize']:
            batchnorm_followed = True
        if 'activation' in items and items['activation'] != 'linear':
            relu_followed = True
        #
        if _section == 'net':
            gen.add_input_layer(items)
        elif _section == 'convolutional':
            gen.add_convolution_layer(items)
            if batchnorm_followed:
                gen.add_batchnorm_layer(items)
                gen.add_scale_layer(items)
            if relu_followed:
                if approximate:
                    gen.add_relu_separate_layer(items) # relu done not in place, but out different blob
                    gen.add_power_layer(items)
                    gen.add_eltwise_layer(items)
                else:
                    gen.add_leaky_relu_layer(items) #assume all pre trained yolo model use leaky
        elif _section == 'connected':
            gen.add_innerproduct_layer(items)
            if relu_followed:
                gen.add_leaky_relu_layer(items) #assume all pre trained yolo model use leaky
        elif _section == 'maxpool':
            gen.add_pooling_layer('MAX', items)
        elif _section == 'avgpool':
            gen.add_pooling_layer('AVE', items, global_pooling=True)
        elif _section == 'dropout':
            gen.add_dropout_layer(items)
        elif _section == 'softmax':
            gen.add_softmax_layer(items)
        else:
            logging.error("{} layer is not supported".format(_section))
    gen.update_last_convolution_layer()
    #gen.finalize('result')
    gen.write(ptxtfile)

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO cfg to Caffe prototxt')
    parser.add_argument('cfg', type=str, help='YOLO cfg')
    parser.add_argument('prototxt', type=str, help='Caffe prototxt')
    parser.add_argument('--approx', help='flag whether to approximate leaky relu or not (for TensorRT implementation',action='store_true')
    args = parser.parse_args()

    convert(args.cfg, args.prototxt, args.approx)

if __name__ == "__main__":
    main()

# vim:sw=4:ts=4:et
