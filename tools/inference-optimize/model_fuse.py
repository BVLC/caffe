#!/usr/bin/env python

import os,sys
import caffe
import numpy as np
#import utils as ut
import csv,math
import subprocess, sys
import string
import copy
import argparse
from google.protobuf import text_format
from pdb import set_trace

def resnet_block_to_fuse_type(model, cur_conv_index):
    maxindex = len(model.layer)-1
    if cur_conv_index+1>maxindex:
        return 0 #UNFUSED
    elif  cur_conv_index+2>maxindex:
        actual = [model.layer[cur_conv_index+1].type, 'xxx', 'xxx', 'xxx']
    elif  cur_conv_index+3>maxindex:
        actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, 'xxx', 'xxx']
    elif  cur_conv_index+4>maxindex:
        actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, model.layer[cur_conv_index+3].type, 'xxx']
    else:
        actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, model.layer[cur_conv_index+3].type, model.layer[cur_conv_index+4].type]
    resnet = ['BatchNorm', 'Scale', 'ReLU']
    resnet_merged = ['ReLU']
    resnet_elt = ['BatchNorm', 'Scale', 'Eltwise', 'ReLU']
    resnet_elt_merged = ['Eltwise', 'ReLU']
    if actual[:1] == resnet_merged:
        return (2, model.layer[cur_conv_index+1].relu_param.negative_slope) #FUSED_CONV_RELU TODO: not magic number
    if actual[:3] == resnet:
        return (2, model.layer[cur_conv_index+3].relu_param.negative_slope) #FUSED_CONV_RELU TODO: not magic number
    if actual[:2] == resnet_elt_merged:
        return (3, model.layer[cur_conv_index+2].relu_param.negative_slope)#FUSED_CONV_ELTWISE_RELU
    if actual == resnet_elt:
        return (3, model.layer[cur_conv_index+4].relu_param.negative_slope) #FUSED_CONV_ELTWISE_RELU
    return 0,0 #UNFUSED

def find_fused_blob_names(model, cur_conv_index):
    i = cur_conv_index + 1
    new_top = None
    elt_bottom = None
    while model.layer[i].type in {'BatchNorm', 'Scale', 'Eltwise', 'ReLU'}:
        if model.layer[i].type == 'Eltwise':
            elt_bottom = model.layer[i].bottom[0]
        i = i + 1
    new_top = model.layer[i].bottom[0]
    return new_top, elt_bottom

def str_to_precision_mode(mode):
    if mode == 'HALF_NONE':
        return 0
    if mode == 'HALF_FLOAT_DATA':
        return 1
    if mode == 'HALF_HALF_DATA':
        return 2
    if mode == 'HALF_ALL':
        return 3

def set_input(in_model, out_model, half_precision_mode):
    out_model.name = in_model.name
    if(half_precision_mode != 'HALF_NONE'):
      out_model.half_precision_mode = str_to_precision_mode(half_precision_mode)
    #For input
    for i in range(len(in_model.input)):
        out_model.input.extend([in_model.input[i]])
        if len(in_model.input_shape) < i:
            out_model.input_shape.extend([in_model.input_shape[i]])
    for i in range(len(in_model.input_dim)):
            out_model.input_dim.extend([in_model.input_dim[i]])

def is_fused_layer(model, layer_index):
    if model.layer[layer_index].type in {'BatchNorm', 'Scale'}:
        return True
    # Fuse with Conv layer.
    elif model.layer[layer_index].type in {'ReLU', 'Eltwise'} and model.layer[layer_index-1].type not in {'InnerProduct'}:
        return True  # Skip ReLU in case of layers fusing
    # Fuse with LRN layer.
    elif model.layer[layer_index].type in {'Pooling'} and model.layer[layer_index-1].type in {'LRN'} and model.layer[layer_index].pooling_param.pool == 0:
        return True
    else:
        return False

def fuse_conv_layer(in_model, in_index, out_model, new_index):
    if out_model.layer[new_index].type == 'Convolution':
        (fuse_mode, negative_slope) = resnet_block_to_fuse_type(in_model, in_index)
        if len(in_model.layer)>in_index+2 and [in_model.layer[in_index+1].type, in_model.layer[in_index+2].type] == ['BatchNorm', 'Scale']:
            out_model.layer[new_index].convolution_param.bias_term = True
        out_model.layer[new_index].convolution_param.fuse_type = fuse_mode
        if fuse_mode == 3:  # FUSED_CONV_ELTWISE_RELU, need to change top name to orig ReLU's top name
            new_top, elt_bottom = find_fused_blob_names(in_model, in_index)
            out_model.layer[new_index].top.remove(out_model.layer[new_index].top[0])
            out_model.layer[new_index].top.append(new_top)
            out_model.layer[new_index].bottom.append(elt_bottom)
        if fuse_mode != 0:
            out_model.layer[new_index].convolution_param.relu_param.negative_slope = negative_slope

def fuse_lrn_layer(in_model, in_index, out_model, out_index):
    if out_model.layer[out_index].type == 'LRN' and in_model.layer[in_index + 1].type == 'Pooling' and in_model.layer[
                in_index + 1].pooling_param.pool == 0:
        new_top = in_model.layer[in_index + 1].top[0]
        out_model.layer[out_index].top.remove(out_model.layer[out_index].top[0])
        out_model.layer[out_index].top.append(new_top)
        out_model.layer[out_index].lrn_param.fuse_type = 1  # 'FUSED_POOL_MAX'
        pooling_param = in_model.layer[in_index + 1].pooling_param
        out_model.layer[out_index].lrn_param.pooling_param.pool = pooling_param.pool
        out_model.layer[out_index].lrn_param.pooling_param.kernel_size.append(pooling_param.kernel_size[0])
        out_model.layer[out_index].lrn_param.pooling_param.stride.append(pooling_param.stride[0])

def set_layers(in_model, out_model):
    out_index = 0
    for in_index in range(0, len(in_model.layer)):
        if is_fused_layer(in_model, in_index):
            continue
        else:
            out_model.layer.extend([in_model.layer[in_index]])
            fuse_conv_layer(in_model, in_index, out_model, out_index)
            fuse_lrn_layer(in_model, in_index, out_model, out_index)
            out_index = out_index + 1

def create_new_model(in_model, half_precision_mode):
    out_model = caffe.proto.caffe_pb2.NetParameter()
    set_input(in_model, out_model, half_precision_mode)
    set_layers(in_model, out_model)
    return out_model

def load_model(filename):
    model = caffe.proto.caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    text_format.Merge(str(input_file.read()), model)
    input_file.close()
    return model

def save_model(model, filename):
    output_file = open(filename, 'w')
    text_format.PrintMessage(model, output_file)
    output_file.close()
   
def find_layerindex_by_name(model, layer_name):
    k = 0
    while model.layer[k].name != layer_name:
        k += 1
        if (k > len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return k

def define_arguments(parser):
    parser.add_argument('--indefinition', type=str,
                        default='deploy.prototxt',
                       help='input network definition (prototxt)')
    parser.add_argument('--inmodel', type=str,
                        default='bvlc_alexnet.caffemodel',
                       help='input network parameters (caffemodel)')
    parser.add_argument('--outdefinition', type=str,
                        default='new_deploy.prototxt',
                       help='output network definition (prototxt)')
    parser.add_argument('--outmodel', type=str,
                        default='new_bvlc_alexnet.caffemodel',
                       help='output network parameters (caffemodel; will be overwritten)')
    parser.add_argument('--half_precision_mode', type = str,
                        default='HALF_NONE',
                        help='float half precision mode')
    parser.add_argument('--fuse_resnet_block', dest='fuse_resnet_block', action='store_true',
                        default=True,
                        help='indicates whether to fuse conv-(batchnorm-scale)-relu block into the conv')
    parser.add_argument('--proto_only', dest='proto_only', action='store_true',
                        default=False,
                        help='indicates whether to generate merged network definition (prototxt) only, without touching the weights')

def parse_args():
    parser = argparse.ArgumentParser(description='convert a network using ' +
       'batch normalization into an equivalent network that does not.  Assumes that ' +
       'parameter layer names have the form \'conv*\' or \'fc*\' and are directly ' +
       'followed by their respective batch norm layers.  These BatchNorm layers must ' +
       'have names which are identical to the names of the layers that they modify, ' +
       'except that \'conv\' or \'fc\' is replaced by \'bn\'.  E.g. conv3 is directly ' +
       'followed by \'bn3\'.  Does not copy fc7, fc8, or fc9.')
    define_arguments(parser)

    args = parser.parse_args()
    return args

def generate_weights(in_model, args):
    in_net = caffe.Net(args.indefinition, args.inmodel ,caffe.TEST)
    #required for working with the fused layers
    caffe.set_device(0)
    caffe.set_mode_gpu()
    out_net =caffe.Net(args.outdefinition,caffe.TEST)
    tocopy=out_net.params

    for prm in tocopy:
      k = find_layerindex_by_name(in_model, prm)
      if in_model.layer[k].type in {'InnerProduct', 'Scale'}:
        for i in range(0,len(in_net.params[prm])):
          out_net.params[prm][i].data[...]=np.copy(in_net.params[prm][i].data[...])
        continue
    #TODO:Need fix conv+bn
      if (in_model.layer[k].type == 'Convolution'): # Assuming convolution is followed by bn and scale
        if k + 1 < len(in_model.layer):
          next1type = in_model.layer[k + 1].type
        else:
          next1type = 'end'
        if k + 2 < len(in_model.layer):
          next2type = in_model.layer[k + 2].type
        else:
          next2type = 'end'
      else:
        print 'Warning: ' + prm + ' has parameters but I can\'t infer its layer type.'
        continue
      if next2type not in {'Scale'}:
        print next2type + ' not found, just ignoring scale ' + prm
        isScale = False
      else:
        isScale = True
        sclprm = in_model.layer[k + 2].name
      if next1type not in {'BatchNorm'}:
        print next1type + ' not found, just copying ' + prm
        for i in range(0,len(in_net.params[prm])):
          out_net.params[prm][i].data[...]=np.copy(in_net.params[prm][i].data[...]);
        continue;
      else:
          bnprm = in_model.layer[k + 1].name
      if in_net.params[prm][0].data.shape != out_net.params[prm][0].data.shape:
        print 'Warning: ' + prm + ' has parameters but they are of different sizes in the different protos.  skipping.'
        continue;
      print 'Removing batchnorm from ' + prm;

      #for i in range(0,len(net2.params[prm])): # first blob for conv layers is the weights, second is the bias. No need for the loop
      i = 0
      if True:
        prmval=np.copy(in_net.params[prm][i].data).reshape(out_net.params[prm][i].data.shape);


        meanval=np.copy(in_net.params[bnprm][0].data);
        stdval=np.copy(in_net.params[bnprm][1].data);
        scaleFactor =np.copy(in_net.params[bnprm][2].data);

        meanval/=in_net.params[bnprm][2].data[...].reshape(-1);
        stdval/=in_net.params[bnprm][2].data[...].reshape(-1);
        eps=None;
        for j in range(0, len(in_model.layer)):
          if str(in_model.layer[j].name) == bnprm:
            eps=in_model.layer[j].batch_norm_param.eps;

        if eps is None:
          raise ValueError("Unable to get epsilon for layer " + nbprm);

        stdval+=eps;

        stdval=np.sqrt(stdval);

        prmval /= stdval.reshape((-1,1,1,1));
        bias1 = -meanval / stdval

        if isScale:
            print 'Removing Scale Layer'
            Scale_Layer_param =np.copy(in_net.params[sclprm][0].data)
            Scale_Layer_param_bias =np.copy(in_net.params[sclprm][1].data)

            Scale_Layer_paramBeta =np.copy(in_net.params[sclprm][1].data)
            prmval= prmval*Scale_Layer_param.reshape((-1,1,1,1))

            mul_bias1_scale = [x * y for x, y in zip(bias1, Scale_Layer_param)]
            bias1  = Scale_Layer_param_bias + mul_bias1_scale

        out_net.params[prm][i].data[:]=prmval
        no_prior_bias = False
        if len(out_net.params[prm]) < 2 : #no bias
            out_net.params[prm].add_blob()
            out_net.params[prm][1].reshape(len(bias1))
            no_prior_bias = True

        if no_prior_bias:
            out_net.params[prm][1].data[:] = bias1
        else:
            out_net.params[prm][1].data[:] = bias1 + out_net.params[prm][1].data[:]

    print 'New caffemodel done'
    out_net.save(args.outmodel);

def generate_prototxt(in_proto, args):
    out_model = create_new_model(in_proto, args.half_precision_mode)
    save_model(out_model, args.outdefinition)
    print 'New proto done'

def generate_new_model(args):
    in_model = load_model(args.indefinition)
    generate_prototxt(in_model, args)
    if not args.proto_only:
        generate_weights(in_model, args)


def main(argv):
    # parse args
    args = parse_args()
    generate_new_model(args)

if __name__ == '__main__':
    main(sys.argv)
