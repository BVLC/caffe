#!/usr/bin/env python

import os,sys
caffe_python = os.path.dirname(os.path.realpath(__file__)) + '/../../python'
sys.path.insert(0, caffe_python)
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

class FuseMode(object):
    UNFUSED = 0
    CONV_MAX_POOLING_RELU = 1
    CONV_RELU = 2
    CONV_ELTWISE_RELU = 3
    CONV_ELTWISE = 4
    CONV_PRELU = 5
    CONV_ELTWISE_PRELU = 6
    CONV_BN_SCALE = 7 # must be the last conv fuse id.
    CONV_FUSE_MAX = 7
    LRN_POOLING_MAX = 11

def is_conv_fusion(mode):
    return mode <= FuseMode.CONV_FUSE_MAX and mode > FuseMode.UNFUSED

def is_lrn_fusion(mode):
    return mode == FuseMode.LRN_POOLING_MAX

def has_relu(mode):
    return mode > FuseMode.UNFUSED and mode < FuseMode.CONV_ELTWISE

def has_prelu(mode):
    return mode >= FuseMode.CONV_PRELU and mode <= FuseMode.CONV_ELTWISE_PRELU

def is_used(in_model, start_index, blob):
    for in_index in range(start_index, len(in_model.layer)):
        for index in range(0, len(in_model.layer[in_index].bottom)):
            #print "check layer " + in_model.layer[in_index].name
            if blob == in_model.layer[in_index].bottom[index]:
                return 1
    return 0

# If any top blob is used in other layers, we have to fallback.
def fixup_fuse_mode(in_model, in_index, fuse_mode, fused_layer_count):
    if fuse_mode == FuseMode.UNFUSED:
        return (fuse_mode, fused_layer_count)
    for index in range(in_index, in_index + fused_layer_count):
        check_list = list();
        for top_index in range(0, len(in_model.layer[index].top)):
            skip = 0;
            for j in range(index + 1, in_index + fused_layer_count + 1):
                for k in range(0, len(in_model.layer[j].top)):
                    if in_model.layer[index].top[top_index] == in_model.layer[j].top[k]:
                        skip = 1
                        break
                if skip:
                   break
            if skip == 0:
              check_list.append(in_model.layer[index].top[top_index])
        for i,blob in enumerate(check_list):
            if is_used(in_model, in_index + fused_layer_count + 2, blob):
               return (0, 0)
    return (fuse_mode, fused_layer_count)

def check_fuse_type(model, cur_index):
    (fuse_mode, fused_layer_count) = (FuseMode.UNFUSED, 0)
    if model.layer[cur_index].type == 'Convolution':
        cur_conv_index = cur_index
        maxindex = len(model.layer)-1
        if cur_conv_index+1>maxindex:
            (fuse_mode, fused_layer_count) = (FuseMode.UNFUSED, 0) #UNFUSED
        elif  cur_conv_index+2>maxindex:
            actual = [model.layer[cur_conv_index+1].type, 'xxx', 'xxx', 'xxx']
        elif  cur_conv_index+3>maxindex:
            actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, 'xxx', 'xxx']
        elif  cur_conv_index+4>maxindex:
            actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, model.layer[cur_conv_index+3].type, 'xxx']
        else:
            actual = [model.layer[cur_conv_index+1].type, model.layer[cur_conv_index+2].type, model.layer[cur_conv_index+3].type, model.layer[cur_conv_index+4].type]
        bn_scale_relu = ['BatchNorm', 'Scale', 'ReLU']
        relu = ['ReLU']
        bn_scale_elt_relu = ['BatchNorm', 'Scale', 'Eltwise', 'ReLU']
        elt_relu = ['Eltwise', 'ReLU']
        bn_scale_elt = ['BatchNorm', 'Scale', 'Eltwise']
        bn_scale_prelu = ['BatchNorm', 'Scale', 'PReLU']
        prelu = ['PReLU']
        bn_scale_elt_prelu = ['BatchNorm', 'Scale', 'Eltwise', 'PReLU']
        elt_prelu = ['Eltwise', 'PReLU']
        bn_scale = ['BatchNorm', 'Scale']
        (fuse_mode, fused_layer_count) = (FuseMode.UNFUSED, 0)
        if actual == bn_scale_elt_relu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_ELTWISE_RELU, 4)
        elif actual[:3] == bn_scale_relu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_RELU, 3)
        elif actual == bn_scale_elt_prelu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_ELTWISE_PRELU, 4)
        elif actual[:3] == bn_scale_prelu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_PRELU, 3)
        elif actual[:3] == bn_scale_elt:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_ELTWISE, 3)
        elif actual[:2] == elt_relu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_ELTWISE_RELU, 2)
        elif actual[:2] == elt_prelu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_ELTWISE_PRELU, 2)
        elif actual[:1] == relu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_RELU, 1)
        elif actual[:1] == prelu:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_PRELU, 1)
        elif actual[:2] == bn_scale:
            (fuse_mode, fused_layer_count) = (FuseMode.CONV_BN_SCALE, 2)
    if model.layer[cur_index].type == 'LRN' and model.layer[cur_index + 1].type == 'Pooling' and model.layer[
       cur_index + 1].pooling_param.pool == 0:
        (fuse_mode, fused_layer_count) = (FuseMode.LRN_POOLING_MAX, 1)
    return fixup_fuse_mode(model, cur_index, fuse_mode, fused_layer_count)

def set_input(in_model, out_model):
    out_model.name = in_model.name
    #For input
    for i in range(len(in_model.input)):
        out_model.input.extend([in_model.input[i]])
        if len(in_model.input_shape) < i:
            out_model.input_shape.extend([in_model.input_shape[i]])
    for i in range(len(in_model.input_dim)):
            out_model.input_dim.extend([in_model.input_dim[i]])

def fuse_layer(in_model, in_index, out_model, new_index):
    (fuse_mode, fused_layer_count) = check_fuse_type(in_model, in_index)
    if is_conv_fusion(fuse_mode):
        if len(in_model.layer)>in_index+2 and [in_model.layer[in_index+1].type, in_model.layer[in_index+2].type] == ['BatchNorm', 'Scale']:
            out_model.layer[new_index].convolution_param.bias_term = True
        # For CONV + BN + SCALE, after we transfer the conv parameters, it is just a normal conv layer for caffe.
        if fuse_mode != FuseMode.CONV_BN_SCALE:
            out_model.layer[new_index].convolution_param.fuse_type = fuse_mode
        new_top = in_model.layer[in_index + fused_layer_count].top[0]
        out_model.layer[new_index].top.remove(out_model.layer[new_index].top[0])
        out_model.layer[new_index].top.append(new_top)
        if fuse_mode == FuseMode.CONV_ELTWISE_RELU or fuse_mode == FuseMode.CONV_ELTWISE_PRELU:
            out_model.layer[new_index].bottom.append(in_model.layer[in_index + fused_layer_count - 1].bottom[0])
        if fuse_mode == FuseMode.CONV_ELTWISE:
            out_model.layer[new_index].bottom.append(in_model.layer[in_index + fused_layer_count].bottom[0])
        if has_relu(fuse_mode) and in_model.layer[in_index + fused_layer_count].relu_param.negative_slope != 0:
            out_model.layer[new_index].convolution_param.relu_param.negative_slope = in_model.layer[in_index + fused_layer_count].relu_param.negative_slope
        if has_prelu(fuse_mode) and in_model.layer[in_index + fused_layer_count].prelu_param.channel_shared != False:
            out_model.layer[new_index].convolution_param.relu_param.negative_slope = in_model.layer[in_index + fused_layer_count].prelu_param.channel_shared
    if is_lrn_fusion(fuse_mode):
        new_top = in_model.layer[in_index + 1].top[0]
        out_model.layer[new_index].top.remove(out_model.layer[new_index].top[0])
        out_model.layer[new_index].top.append(new_top)
        out_model.layer[new_index].lrn_param.fuse_type = 1  # 'FUSED_POOL_MAX'
        pooling_param = in_model.layer[in_index + 1].pooling_param
        out_model.layer[new_index].lrn_param.pooling_param.pool = pooling_param.pool
        out_model.layer[new_index].lrn_param.pooling_param.kernel_size.append(pooling_param.kernel_size[0])
        out_model.layer[new_index].lrn_param.pooling_param.stride.append(pooling_param.stride[0])
    return fused_layer_count

def set_layers(in_model, out_model):
    out_index = 0
    next_in_index = 0
    for in_index in range(0, len(in_model.layer)):
        if in_index != next_in_index:
            continue
        out_model.layer.extend([in_model.layer[in_index]])
        step = fuse_layer(in_model, in_index, out_model, out_index)
        out_index = out_index + 1;
        next_in_index = in_index + step + 1;

def create_new_model(in_model):
    out_model = caffe.proto.caffe_pb2.NetParameter()
    set_input(in_model, out_model)
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
    out_net =caffe.Net(args.outdefinition,caffe.TEST)
    tocopy=out_net.params

    param_list = list()

    for prm in tocopy:
        k = find_layerindex_by_name(in_model, prm)
        param_list.append((k, prm))

    param_list.sort()

    for i,(k,prm) in enumerate(param_list):
        (fuse_mode, fused_layer_count) = check_fuse_type(in_model, k)
        if in_model.layer[k + fused_layer_count].type in {'PReLU'}:
            isPrelu = True
            prelu_prm = in_model.layer[k + fused_layer_count].name

        if fuse_mode == FuseMode.UNFUSED or fuse_mode == FuseMode.LRN_POOLING_MAX \
           or (fuse_mode == FuseMode.CONV_ELTWISE and fused_layer_count == 2)     \
           or ((fuse_mode == FuseMode.CONV_RELU or fuse_mode == FuseMode.CONV_PRELU) and fused_layer_count == 1):
            print '    Copying ' + prm
            for i in range(0,len(in_net.params[prm])):
                out_net.params[prm][i].data[...]=np.copy(in_net.params[prm][i].data[...])
            if has_prelu(fuse_mode):
                if not isPrelu:
                    print 'Incorrect fusion detected for Prelu, aborting'
                    exit(-1)
                out_net.params[prm].add_blob()
                out_net.params[prm][-1].reshape(in_net.params[prelu_prm][0].shape[0])
                out_net.params[prm][-1].data[:] = in_net.params[prelu_prm][0].data[:]
            continue
        print '    Processing ' + prm + ' with fusion mode %r, fusing the following layers' % fuse_mode
        for i in range(0, fused_layer_count):
            print '        ' + in_model.layer[k + 1 + i].name
        if in_model.layer[k + 1].type in {'BatchNorm'}:
            bnprm = in_model.layer[k + 1].name

        if in_model.layer[k + 2].type in {"Scale"}:
            isScale = True
            sclprm = in_model.layer[k + 2].name

        if in_net.params[prm][0].data.shape != out_net.params[prm][0].data.shape:
            print '    Warning: ' + prm + ' has parameters but they are of different sizes in the different protos.  skipping.'
            continue;
        print '    Removing batchnorm from ' + prm;

        prmval=np.copy(in_net.params[prm][0].data).reshape(out_net.params[prm][0].data.shape);

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
            raise ValueError("Unable to get epsilon for layer " + bnprm);

        stdval+=eps;

        stdval=np.sqrt(stdval);

        prmval /= stdval.reshape((-1,1,1,1));
        bias1 = -meanval / stdval

        if isScale:
            print '    Removing Scale Layer'
            Scale_Layer_param =np.copy(in_net.params[sclprm][0].data)
            Scale_Layer_param_bias =np.copy(in_net.params[sclprm][1].data)

            Scale_Layer_paramBeta =np.copy(in_net.params[sclprm][1].data)
            prmval= prmval*Scale_Layer_param.reshape((-1,1,1,1))

            mul_bias1_scale = [x * y for x, y in zip(bias1, Scale_Layer_param)]
            bias1  = Scale_Layer_param_bias + mul_bias1_scale

        out_net.params[prm][0].data[:]=prmval
        no_prior_bias = False
        if len(out_net.params[prm]) < 2 : #no bias
            out_net.params[prm].add_blob()
            out_net.params[prm][1].reshape(len(bias1))
            no_prior_bias = True
        if no_prior_bias:
            out_net.params[prm][1].data[:] = bias1
        else:
            out_net.params[prm][1].data[:] = bias1 + out_net.params[prm][1].data[:]
        if has_prelu(fuse_mode):
            if not isPrelu:
                print 'Incorrect fusion detected for Prelu, aborting'
                exit(-1)
            out_net.params[prm].add_blob()
            out_net.params[prm][-1].reshape(in_net.params[prelu_prm][0].shape[0])
            out_net.params[prm][-1].data[:] = in_net.params[prelu_prm][0].data[:]
    print 'New caffemodel generated successfully.'
    out_net.save(args.outmodel);

def generate_prototxt(in_proto, args):
    out_model = create_new_model(in_proto)
    save_model(out_model, args.outdefinition)
    print 'New proto generated successfully.'

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
