#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2018 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import sys
import copy
import argparse
pycaffe = os.path.split(os.path.realpath(__file__))[0] + '/../python'
sys.path.insert(0, pycaffe)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import sampling
import numpy as np

int8_layers = ["Convolution", "ReLU", "Split", "Concat", "Pooling", "Eltwise", "InnerProduct"]
quantize_layers = ["Convolution", "InnerProduct"]

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_existence(path):
    try:
        return os.path.exists(path)
    except Exception as e:
        raise ("Failed to check {} existence due to {}".format(path, str(e)))


def setup_env():
    os.chdir(os.path.dirname(os.path.abspath(params.root)))
    caffe.set_mode_cpu()


def read_prototxt(prototxt):
    try:
        if not check_existence(prototxt):
            return None
        net = caffe_pb2.NetParameter()
        with open(prototxt) as f:
            txtf.Merge(f.read(), net)

        return net

    except Exception as e:
        print ("Failed to read {} due to {}".format(prototxt, e))


def get_input_layers(l, net, end):
    top_layers = []
    for layerIndex in range(0, end):
        reverse_layer_index = end - layerIndex - 1
        for blobIndex in range(0, len(net.layer[reverse_layer_index].top)):
            if net.layer[reverse_layer_index].top[blobIndex] in l.bottom:
                top_layers.append((reverse_layer_index, net.layer[reverse_layer_index].name,
                                   net.layer[reverse_layer_index].type))
    return top_layers


def get_input_convolutions(l, net, end, interesting_layers, uninteresting_layers=[]):
    all_input_layers = []
    input_layers = get_input_layers(l, net, end)
    while True:
        if len(input_layers) == 0:
            break

        processed_layers = input_layers  # sync inputLayers change
        for lp in processed_layers:
            if lp[2] not in int8_layers:
                input_layers.remove(lp)
                continue
            if lp[2] in interesting_layers:
                input_layers.remove(lp)
                if lp not in all_input_layers:
                    all_input_layers.append(lp)
                continue

            if lp[2] not in uninteresting_layers:
                new_input_layers = get_input_layers(net.layer[lp[0]], net, lp[0])
                input_layers.remove(lp)
                input_layers.extend(new_input_layers)
            else:
                input_layers.remove(lp)

    return all_input_layers

def analyze_conv_output_with_relu(compiled_net):
    convs_output_with_relu = []
    for _, layer in enumerate(compiled_net.layer):
        if layer.type == 'Convolution':
            if layer.convolution_param.relu and isclose(layer.convolution_param.negative_slope, 0.0):
                convs_output_with_relu.append(layer.name)
    return convs_output_with_relu


def analyze_conv_input_u8(conv_inputs, convs_output_with_relu):
    for conv_input in conv_inputs:
        if conv_input[1] not in convs_output_with_relu:
            return False
    return True
    

def find_index_by_name(name, layer_infos):
    for (l, index) in layer_infos:
        if name == l.name:
            return index
    return -1


def is_convolution_input_u8(l, net, end, interesting_layers, convs_output_with_relu):
    all_input_layers = []
    input_layers = get_input_layers(l, net, end)
    while True:
        if len(input_layers) == 0:
            break

        for input_layer in input_layers:
            if input_layer[2] in quantize_layers and (input_layer[1] not in convs_output_with_relu):
                return False

        processed_layers = input_layers  # sync inputLayers change
        for lp in processed_layers:
            if lp[2] not in int8_layers:
                input_layers.remove(lp)
                continue
            if lp[2] in interesting_layers:
                input_layers.remove(lp)
                if lp not in all_input_layers:
                    all_input_layers.append(lp)
                continue

            new_input_layers = get_input_layers(net.layer[lp[0]], net, lp[0])
            input_layers.remove(lp)
            input_layers.extend(new_input_layers)
    
    return True

def analyze_conv_output_with_relu_from_net(convs_output_with_relu, compiled_net, net):
    new_convs_output_with_relu = []
    new_convs_output_with_relu.extend(convs_output_with_relu)

    compiled_relu_layers = [(value, index) for index, value in enumerate(compiled_net.layer) if value.type == 'ReLU' and value.relu_param.negative_slope != 0]
    if len(compiled_relu_layers) != 0:
        relu_layers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'ReLU']
        for (l, index) in relu_layers:
            conv_inputs = get_input_convolutions(l, net, index, ["Convolution"]) # FIXME
            new_convs_output_with_relu.append(conv_inputs[0][1])

    return new_convs_output_with_relu

def find_last_conv_layers(top_blobs_map, bottom_blobs_map, convolution_layers, non_int8_layers):
    last_conv_indexes = []
    for (l, index) in convolution_layers:
        top_blob = top_blobs_map[l.name][0]
        for k, v in bottom_blobs_map.iteritems():
            if len(v) == 1 and v[0] == top_blob:
                output_index = find_index_by_name(k, non_int8_layers)
                if output_index != -1:
                    last_conv_indexes.append(index)

    return last_conv_indexes

def transform_convolutions(model_path, compiled_model_path, top_blobs_map, bottom_blobs_map, use_unsigned_range, concat_use_fp32, unify_concat_scales, conv_algo, enable_1st_conv = False):
    net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, net)

    compiled_net = caffe_pb2.NetParameter()
    with open(compiled_model_path) as f:
        s = f.read()
        txtf.Merge(s, compiled_net)

    convs_output_with_relu = analyze_conv_output_with_relu(compiled_net)
    # extended convs output with relu is used for convs that cannot fuse with relu due to negative slope
    # extended_convs_output_with_relu = analyze_conv_output_with_relu_from_net(convs_output_with_relu, compiled_net, net)
    new_net = copy.deepcopy(net)

    convolution_layers = [(value, index) for index, value in enumerate(net.layer) if value.type in quantize_layers]
    compiled_convolution_layers = [(value, index) for index, value in enumerate(compiled_net.layer) if value.type in quantize_layers]

    u8_max = 255
    s8_max = 127
    first_conv = True if enable_1st_conv else False
    for (l, index) in convolution_layers:
        for si in range(0, len(new_net.layer[index].quantization_param.scale_out)):
            if l.name in convs_output_with_relu:  # u8
                new_net.layer[index].quantization_param.scale_out[si] = u8_max / new_net.layer[index].quantization_param.scale_out[si]
            else:  # s8
                if use_unsigned_range:
                    new_net.layer[index].quantization_param.scale_out[si] = u8_max / new_net.layer[index].quantization_param.scale_out[si]
                else:
                    new_net.layer[index].quantization_param.scale_out[si] = s8_max / new_net.layer[index].quantization_param.scale_out[si]


        index_in_compiled_net = find_index_by_name(l.name, compiled_convolution_layers)
        assert(index_in_compiled_net >= 0)
        #conv_inputs = get_input_convolutions(l, compiled_net, index_in_compiled_net, ["Convolution"])
        #conv_input_u8 = analyze_conv_input_u8(conv_inputs, convs_output_with_relu)
        conv_input_u8 = is_convolution_input_u8(l, compiled_net, index_in_compiled_net, ["Convolution"], convs_output_with_relu) # FIXME: extended_convs_output_with_relu
        for si in range(0, len(new_net.layer[index].quantization_param.scale_in)):
            if conv_input_u8:  # u8
                if first_conv:
                    new_net.layer[index].quantization_param.scale_in[si] = s8_max / new_net.layer[index].quantization_param.scale_in[si]
                    new_net.layer[index].quantization_param.is_negative_input = True
                    first_conv = False
                else:
                    new_net.layer[index].quantization_param.scale_in[si] = u8_max / new_net.layer[index].quantization_param.scale_in[si]
            else:
                new_net.layer[index].quantization_param.scale_in[si] = s8_max / new_net.layer[index].quantization_param.scale_in[si]
                new_net.layer[index].quantization_param.is_negative_input = True

        for si in range(0, len(new_net.layer[index].quantization_param.scale_params)):
            if not isclose(new_net.layer[index].quantization_param.scale_params[si], 0.0):
                new_scale_param = s8_max / new_net.layer[index].quantization_param.scale_params[si]
                if np.isinf(new_scale_param):
                    new_scale_param = 0.0
                new_net.layer[index].quantization_param.scale_params[si] = new_scale_param
            else:
                new_net.layer[index].quantization_param.scale_params[si] = 0.0

        if conv_algo:
            for conv_input in conv_inputs:
                index_bottom_layer = find_index_by_name(conv_input[1], convolution_layers)
                for si in range(0, len(new_net.layer[index_bottom_layer].quantization_param.scale_out)):
                    new_net.layer[index_bottom_layer].quantization_param.scale_out[si]  = new_net.layer[index].quantization_param.scale_in[si]

    concat_layers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'Concat']
    if len(concat_layers) > 0:
        compiled_concat_layers = [(value, index) for index, value in enumerate(compiled_net.layer) if value.type == 'Concat']
        concat_layers.reverse()
        if unify_concat_scales:
            for (l, index) in concat_layers:
                index_in_compiled_net = find_index_by_name(l.name, compiled_concat_layers)
                assert(index_in_compiled_net >= 0)
                conv_inputs = get_input_convolutions(l, compiled_net, index_in_compiled_net, ["Convolution"], ["Concat"])
                # TODO: support resonable cross-levels concat scale unify
                min_concat_scale = sys.float_info.max
                concat_input_indexes = []
                for conv_input in conv_inputs:
                    index_in_net = find_index_by_name(conv_input[1], convolution_layers)
                    assert(index_in_net >= 0)
                    concat_input_indexes.append(index_in_net)
                    if new_net.layer[index_in_net].quantization_param.scale_out[0] < min_concat_scale:
                        min_concat_scale = new_net.layer[index_in_net].quantization_param.scale_out[0]

                for concat_input_index in concat_input_indexes:
                    new_net.layer[concat_input_index].quantization_param.scale_out[0] = min_concat_scale
        else:
            if concat_use_fp32:
                for (l, index) in concat_layers:
                    index_in_compiled_net = find_index_by_name(l.name, compiled_concat_layers)
                    assert(index_in_compiled_net >= 0)
                    conv_inputs = get_input_convolutions(l, compiled_net, index_in_compiled_net, ["Convolution"])
                    for conv_input in conv_inputs:
                        index_in_net = find_index_by_name(conv_input[1], convolution_layers)
                        assert(index_in_net >= 0)
                        new_net.layer[index_in_net].quantization_param.bw_layer_out = 32
                        new_net.layer[index_in_net].quantization_param.scale_out[:] = [1.0]

    non_int8_layers = [(value, index) for index, value in enumerate(net.layer) if value.type not in int8_layers]
    last_conv_indexes = find_last_conv_layers(top_blobs_map, bottom_blobs_map, convolution_layers, non_int8_layers)
    
    # TODO: support last convolution without consumer
    for index in last_conv_indexes:
        new_net.layer[index].quantization_param.bw_layer_out = 32
        new_net.layer[index].quantization_param.scale_out[:] = [1.0]

    with open(model_path, 'w') as f:
        f.write(str(new_net))


def generate_sample_bak(sample_path, input_model, weights,
                    quantized_model, detection, scaling_mode, iterations=1, error_margin=1):
    cmd = '{0} quantize -model {1} -weights {2} -model_quantized {3} -iterations {4} -error_margin {5} ' \
          ' -scaling {6} -trimming_mode dynamic_fixed_point'.format(sample_path, input_model, weights, quantized_model,
                                                                    iterations, error_margin, scaling_mode)
    if detection:
        cmd += ' --detection=1'
    
    os.system(cmd)


def generate_sample(input_model, weights, quantized_model, scaling_mode, calibration_algo, conv_algo, iterations=10, enable_1st_conv=False):
    (blobs, params, top_blobs_map, bottom_blobs_map, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, winograd_convolutions) = sampling.sample(input_model, weights, conv_algo, iterations, enable_1st_conv)

    (inputs_max, outputs_max, inputs_min) = sampling.calibrate_activations(blobs, conv_top_blob_layer_map, conv_bottom_blob_layer_map, winograd_bottoms, calibration_algo, "SINGLE", conv_algo)
    params_max = sampling.calibrate_parameters(params, winograd_convolutions, "DIRECT", scaling_mode.upper(), conv_algo)

    generate_sample_impl(input_model, quantized_model, inputs_max, outputs_max, inputs_min, params_max, enable_1st_conv)

    return (top_blobs_map, bottom_blobs_map)
 
   
def generate_sample_impl(input_model, quantized_model, inputs_max, outputs_max, inputs_min, params_max, enable_1st_conv=False):
    net = caffe_pb2.NetParameter()
    with open(input_model) as f:
        s = f.read()
        txtf.Merge(s, net)

    new_net = copy.deepcopy(net)
    convolution_layers = [(value, index) for index, value in enumerate(net.layer) if value.type in quantize_layers]
    first_conv = False if enable_1st_conv else True
    for (l, index) in convolution_layers:
        if first_conv:
            first_conv = False
            continue
        new_net.layer[index].quantization_param.bw_layer_in = 8
        new_net.layer[index].quantization_param.bw_layer_out = 8
        new_net.layer[index].quantization_param.bw_params = 8
        new_net.layer[index].quantization_param.scale_in[:] = inputs_max[l.name]
        new_net.layer[index].quantization_param.scale_out[:] = outputs_max[l.name]
        new_net.layer[index].quantization_param.scale_params[:] = params_max[l.name]

    with open(quantized_model, 'w') as f:
        f.write(str(new_net))

def get_compiled_net(caffe_bin, model_def, model_weights, detection):
    output_log_name = '.compiled_net.txt'

    cmd = '{} test -model {} -weights {} -iterations 1 -sampling'.format(caffe_bin, model_def, model_weights)
    if detection:
        cmd += ' -detection'
    cmd += ' 2>&1 > {}'.format(output_log_name)
    os.environ['GLOG_minloglevel'] = '2'
    os.system(cmd)
    os.environ.pop('GLOG_minloglevel')
    return os.path.abspath(output_log_name)


def get_the_accuracy(caffe_bin, model_def, model_weights, iterations, detection, blob_name):
    output_log_name = 'calibrator_log.txt'
    cmd = '{} test -model {} -weights {} -iterations {}'.format(caffe_bin, model_def, model_weights, iterations)
    if detection:
        cmd += ' -detection'
    cmd += ' 2>&1|tee {}'.format(output_log_name)

    os.system(cmd)

    with open(output_log_name) as f:
        data = f.readlines()

    for i in data[::-1]:
        if i.find('{} = '.format(blob_name)) != -1:
            try:
                splits = i.split('=')
                last_split = splits[-1].strip().split(' ')[0] # accuracy or loss
                return float(last_split)
            except Exception as e:
                print 'Failed to generate accuracy due to {}'.format(str(e))
                sys.exit(-1)

    print 'Failed to get accuracy, please check the parameters and rerun the scripts.'
    sys.exit(-1)


def remove_top_quantized_parameter(current_quantized_file):
    net = read_prototxt(current_quantized_file)
    for i in net.layer:
        if i.type == 'Convolution' and i.HasField('quantization_param'):
            i.ClearField('quantization_param')
            break

    with open(current_quantized_file, 'w') as f:
        f.write(str(net))


def tuning_quantized_topology(base_top1_accuracy, prototxt, caffe_bin, model_weights, top_blob_map, bottom_blobs_map, iterations,
                              accuracy_loss, detection, blob_name, quantize_only, use_unsigned_range, 
                              concat_use_fp32, unify_concat_scales, conv_algo, enable_1st_conv):
    print 'Updating quantization parameter...'

    transform_convolutions(prototxt, get_compiled_net(caffe_bin, prototxt, model_weights, detection), top_blobs_map, bottom_blobs_map,
                           use_unsigned_range, concat_use_fp32, unify_concat_scales, conv_algo, enable_1st_conv)
    if quantize_only:
        return

    current_top1_accuracy = get_the_accuracy(caffe_bin, prototxt, model_weights, iterations, detection, blob_name)
    #while abs(current_top1_accuracy - base_top1_accuracy) >= accuracy_loss:
    #    print 'Tuning... '
    #    print abs(current_top1_accuracy - base_top1_accuracy)
    #    remove_top_quantized_parameter(prototxt)
    #    current_top1_accuracy = get_the_accuracy(caffe_bin, prototxt, model_weights, iterations, detection, blob_name)


def accuracy_blob_name_parser(prototxt):
    net = read_prototxt(prototxt)
    if not net:
        print 'Please check the model prototxt integrity.'
        sys.exit(-1)
    res = {}
    for i in net.layer:
        if i.type == 'Accuracy':
            if i.HasField('accuracy_param'):
                res[i.accuracy_param.top_k] = i.top[0]
            else:
                res[1] = i.top[0]
    return res[sorted(res.keys())[0]] if res else ''

def check_blob_name_existence(prototxt, blob_name):
    net = read_prototxt(prototxt)
    if not net.layer:
        print 'Please check the model prototxt integrity.'
        sys.exit(-1)

    for i in net.layer[::-1]:
        for _, value in enumerate(i.top):
            if value == blob_name:
                return True
    return False

def generate_dummy_model(model_path, dummy):
    net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, net)

    first_conv = True
    convolution_layers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'Convolution']
    for (l, index) in convolution_layers:
        if first_conv:
            first_conv = False
            continue
        net.layer[index].quantization_param.bw_layer_in = 8
        net.layer[index].quantization_param.bw_layer_out = 8
        net.layer[index].quantization_param.bw_params = 8
        net.layer[index].quantization_param.scale_in[:] = [1.0]
        net.layer[index].quantization_param.scale_out[:] = [1.0]
        net.layer[index].quantization_param.scale_params[:] = [1.0]

    with open(dummy, 'w') as f:
        f.write(str(net))

if __name__ == '__main__':
    usage_string = 'Usage: 1.Build the caffe\n ' \
                    '2.cd /path/to/caffe/scripts\n ' \
                    '3.python calibrator.py ' \
                    ' -r /path/to/caffe/build ' \
                    ' -w pre-trained-fp32 weights ' \
                    ' -m typology ' \
                    ' -i iterations ' \
                    ' -l acceptable accuracy loss value, the default value is 0.01 stands for one percent' \
                    ' -d 1(0 means classification while 1 means detection, the default value is 0' \
                    ' -n blob name which means accuracy' \
                    ' -c scaling mode, the default value is single' \
                    ' -s sampling iterations'

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help=usage_string)

    parser.add_argument('-i', '--iterations', action='store', dest='iterations', default=10,
                        help='equal to the number to complete one epoch.')

    parser.add_argument('-w', '--weights', action='store', dest='weights', default='',
                        help='pre-trained-fp32-weights.')

    parser.add_argument('-m', '--model', action='store', dest='model', default='',
                        help='topology definition prototxt.')

    parser.add_argument('-l', '--accuracy_loss', action='store', dest='loss', default=0.01,
                        help='the acceptable accuracy loss that raised by 8-Bit quantization, '
                             'default value is 0.01(one percent).')

    parser.add_argument('-d', '--detection', action='store', dest='is_detection', default=0,
                        help='0 for classification while 1 for detection, default value is 0.')

    parser.add_argument('-r', '--root', action='store', dest='root', default='',
                        help='caffe build path')

    parser.add_argument('-n', '--blob_name', action='store', dest='blob_name', default='',
                        help='top blob name which stands for accuracy')

    parser.add_argument('-c', '--weights_channel', action='store', dest='scaling_mode', default='single',
                        help='the scaling mode for weights')
    
    parser.add_argument('-s', '--sampling_iterations', action='store', dest='sampling_iterations', default=10,
                        help='iteration number of sampling, the default value is 10.')

    parser.add_argument('-p', '--performance_model', dest='performance_model', action="store_true", default=False,
                        help='to generate model to measure performance only')

    parser.add_argument('-q', '--quantize_model', dest='quantize_model', action="store_true", default=False,
                        help='to quantize the model only')

    parser.add_argument('-u', '--unsigned_range', dest='unsigned_range', action="store_true", default=False,
                        help='to quantize using unsigned range for activation')

    parser.add_argument('-t', '--concat_use_fp32', dest='concat_use_fp32', action="store_true", default=False,
                        help='to use fp32 for concat')

    parser.add_argument('-f', '--unify_concat_scales', dest='unify_concat_scales', action="store_true", default=False,
                        help='to unify concat scales')

    parser.add_argument('-a', '--calibration_algos', dest='calibration_algos', action='store', default="DIRECT",
                        help='to choose the calibration alogorithm')

    parser.add_argument('-wi', '--conv_algo', dest='conv_algo', action="store_true", default=False,
                        help='to choose the convolution algorithm')

    parser.add_argument('-1st', '--enable_1st_conv', dest='enable_1st_conv', action="store_true", default=False,
                        help='to enable 1st conv quantization')
    
    params = parser.parse_args()
    
    if not check_existence(params.root):
        print 'Please check the {} existence.'.format(params.root)
        sys.exit(-1)

    pycaffe_path = os.path.abspath(os.path.dirname(os.path.abspath(params.root))) + os.path.sep + 'python'
    if not check_existence(pycaffe_path):
        print "Please check the pycaffe existence.Suggest to rebuild pycaffe via 'make pycaffe'"
    sys.path.insert(0, pycaffe_path)
    import caffe
    from caffe.proto import caffe_pb2

    model = os.path.abspath(params.model)
    if not check_existence(model):
        print 'Please check model: {} existence.'.format(model)
        sys.exit(-1)

    dummy_prototxt = model.rsplit('.')[0] + '_dummy.prototxt'
    if params.performance_model:
        generate_dummy_model(model, dummy_prototxt)
        print 'Updated prototxt {} is generated.'.format(dummy_prototxt)
        sys.exit(0)

    try:
        user_input_iterations = int(params.iterations)
    except:
        print 'Set the iterations to the default value 1000'
        user_input_iterations = 1000
    else:
        if user_input_iterations < 1:
            print 'Invalid iterations!The value should be larger than zero.'
            sys.exit(-1)
    try:
        user_sampling_iteration = int(params.sampling_iterations)
    except:
        print 'Set the sampling iteration to the default value 10'
        user_sampling_iteration = 10
    else:
        if user_sampling_iteration < 1:
            print 'Invalid sampling iteration!The value should be larger than zero.'
            sys.exit(-1)

    if params.scaling_mode != 'multiple' and params.scaling_mode != 'single':
        user_scaling_mode = 'single'
    else:
        user_scaling_mode = params.scaling_mode

    if params.calibration_algos != 'DIRECT' and params.calibration_algos != "KL" and params.calibration_algos != "MAXP":
        user_calibration_algos = 'DIRECT'
    else:
        user_calibration_algos = params.calibration_algos

    if params.conv_algo != False and params.conv_algo != True:
        user_conv_algo = False
    else:
        user_conv_algo = params.conv_algo 

    try:
        toleration = float(params.loss)
        if toleration >= 1 or toleration < 0:
            toleration = 0.01
    except:
        print 'Set the toleration to 1%.'
        toleration = 0.01

    try:
        detection_flag = 1 if int(params.is_detection) == 1 else 0
    except:
        print 'Set the test type to classification.'
        detection_flag = 0

    user_enable_1st_conv = params.enable_1st_conv
    user_input_weights = os.path.abspath(params.weights)
    if not check_existence(user_input_weights):
        print 'Please check weights: {} existence.'.format(user_input_weights)
        sys.exit(-1)

    sample = os.path.abspath(params.root + os.path.sep + 'tools/sample')
    if not check_existence(sample):
        print 'Please check sample: {} existence.'.format(sample)
        sys.exit(-1)

    caffe_bin_path = os.path.abspath(params.root + os.path.sep + 'tools/caffe')
    if not check_existence(caffe_bin_path):
        print 'Please check model/weights/sample existence.'
        sys.exit(-1)

    setup_env()

    target_blob_name = params.blob_name
    if not params.quantize_model:
        if not target_blob_name and not detection_flag:
            target_blob_name = accuracy_blob_name_parser(model)

        if not target_blob_name or not check_blob_name_existence(model, target_blob_name):
            print 'Please specify valid blob name and rerun the script.'
            sys.exit(-1)

    quantized_prototxt = model.rsplit('.')[0] + '_quantized.prototxt'
    print 'Sampling...'
    (top_blobs_map, bottom_blobs_map) = generate_sample(model, user_input_weights, quantized_prototxt, user_scaling_mode, user_calibration_algos, user_conv_algo, user_sampling_iteration, user_enable_1st_conv)
    print 'Sampling done'
    top_1 = None
    if not params.quantize_model:
        print 'Generating the FP32 accuracy...'
        top_1 = get_the_accuracy(caffe_bin_path, model, user_input_weights, user_input_iterations, detection_flag,
                             target_blob_name)
        print 'FP32 accuracy is: {}'.format(top_1)

    tuning_quantized_topology(top_1, quantized_prototxt, caffe_bin_path, user_input_weights, top_blobs_map, bottom_blobs_map, user_input_iterations,
                              toleration, detection_flag, target_blob_name, params.quantize_model, params.unsigned_range,
                              params.concat_use_fp32, params.unify_concat_scales, params.conv_algo, user_enable_1st_conv)

    print 'Updated prototxt {} is generated.'.format(quantized_prototxt)
