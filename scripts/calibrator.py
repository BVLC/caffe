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

caffe_root = "../"
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


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
        raise ("Failed to read {} due to {}".format(prototxt, str(e)))


def get_bottom_layers(layer_name, net, start):
    bottom_layers = []
    for index, value in enumerate(net.layer[start:]):
        for sub_index, sub_value in enumerate(value.bottom):
            if sub_value == layer_name:
                bottom_layers.append((index, value.name, value.type))

    return bottom_layers


def get_top_layers(l, net, end):
    top_layers = []
    for layerIndex in range(0, end):
        reverse_layer_index = end - layerIndex - 1
        for blobIndex in range(0, len(net.layer[reverse_layer_index].top)):
            if net.layer[reverse_layer_index].top[blobIndex] in l.bottom:
                top_layers.append((reverse_layer_index, net.layer[reverse_layer_index].name,
                                   net.layer[reverse_layer_index].type))
    return top_layers


def get_all_top_layers(l, net, end, skip_layers, interesting_layers):
    all_top_layers = []
    top_layers = get_top_layers(l, net, end)
    while True:
        if len(top_layers) == 0:
            break

        processed_layers = top_layers  # sync topLayers change
        for (li, ln, lt) in processed_layers:
            if lt in skip_layers:
                top_layers.remove((li, ln, lt))
                continue
            if lt in interesting_layers:
                lp = (li, ln, lt)
                top_layers.remove(lp)
                if lp not in all_top_layers:
                    all_top_layers.append(lp)
                continue

            new_top_layers = get_top_layers(net.layer[li], net, li)
            top_layers.remove((li, ln, lt))
            top_layers.extend(new_top_layers)

    return all_top_layers


def get_all_bottom_layers(layer_name, net, start, skip_layers, interesting_layers):
    all_bottom_layers = []
    bottom_layers = get_bottom_layers(layer_name, net, start)
    while True:
        if len(bottom_layers) == 0:
            break

        processed_layers = bottom_layers  # sync bottom_layers change
        for (li, ln, lt) in processed_layers:
            if lt in skip_layers:
                bottom_layers.remove((li, ln, lt))
                continue
            if lt in interesting_layers:
                lp = (li, ln, lt)
                bottom_layers.remove(lp)
                if lp not in all_bottom_layers:
                    all_bottom_layers.append(lp)
                continue

            new_bottom_layers = get_bottom_layers(ln, net, li + 1)
            bottom_layers.remove((li, ln, lt))
            bottom_layers.extend(new_bottom_layers)

    return all_bottom_layers


def transform_convolutions(model_path):
    net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, net)

    new_net = copy.deepcopy(net)

    convolution_layers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'Convolution']

    interesting_layers = ['ReLU']
    skip_layers = ['Convolution', 'Eltwise', 'Concat']

    u8_max = 255
    s8_max = 127

    for (l, index) in convolution_layers:
        outputwith_relu = get_all_bottom_layers(l.name, net, index + 1, skip_layers, interesting_layers)
        inputwith_relu = get_all_top_layers(l, net, index, skip_layers, interesting_layers)
        # print "Processing", l.type, l.name

        # output_type = 'u8' if outputwith_relu else 's8'
        # input_type = 'u8' if inputwith_relu else 's8'

        for si in range(0, len(new_net.layer[index].quantization_param.scale_out)):
            if len(outputwith_relu) > 0:  # u8
                new_net.layer[index].quantization_param.scale_out[si] = round(u8_max / new_net.layer[index].
                                                                              quantization_param.scale_out[si], 2)
            else:  # s8
                new_net.layer[index].quantization_param.scale_out[si] = round(s8_max / new_net.layer[index].
                                                                              quantization_param.scale_out[si], 2)

        for si in range(0, len(new_net.layer[index].quantization_param.scale_in)):
            if len(inputwith_relu) > 0:  # u8
                new_net.layer[index].quantization_param.scale_in[si] = round(u8_max / new_net.layer[index].
                                                                             quantization_param.scale_in[si], 2)
            else:  # s8
                new_net.layer[index].quantization_param.scale_in[si] = round(s8_max / new_net.layer[index].
                                                                             quantization_param.scale_in[si], 2)

        for si in range(0, len(new_net.layer[index].quantization_param.scale_params)):
            new_net.layer[index].quantization_param.scale_params[si] = round(s8_max / new_net.layer[index].
                                                                             quantization_param.scale_params[si], 2)

    with open(model_path, 'w') as f:
        f.write(str(new_net))


def generate_sample(sample_path, input_model, weights,
                    quantized_model, model_type, iterations=1, error_margin=1, power=0):
    cmd = '{0} quantize -model {1} -weights {2} -model_quantized {3} -iterations {4} ' \
          '-trimming_mode dynamic_fixed_point -error_margin {5} -power {6}'.format(sample_path, input_model, weights,
                                                                                   quantized_model, iterations,
                                                                                   error_margin, power)
    if model_type == 3:
        cmd += ' --detection=1'

    os.system(cmd)


def get_the_accuracy(caffe_bin, model_def, model_weights, iterations, model_type):
    output_log_name = 'calibrator_log.txt'
    cmd = '{} test -model {} -weights {} -iterations {}'.format(caffe_bin, model_def, model_weights, iterations)
    if model_type == 3:
        cmd += ' -detection'
    cmd += ' 2>&1|tee {}'.format(output_log_name)
    os.system(cmd)
    with open(output_log_name) as f:
        data = f.readlines()
    try:
        if model_type == 1:
            top_1 = data[-2].strip()
            return float(top_1.split('=')[-1].strip())
        elif model_type == 2:
            top_1 = data[-3].strip()
            return float(top_1.split('=')[-1].strip())
        elif model_type == 3:
            top_1 = data[-1].strip()
            return float(top_1.split('=')[-1].strip())
    except Exception as e:
        print 'Failed to generate accuracy due to {}'.format(str(e))
        sys.exit(-1)


def remove_top_quantized_parameter(current_quantized_file):
    net = read_prototxt(current_quantized_file)
    for i in net.layer:
        if i.type == 'Convolution' and i.HasField('quantization_param'):
            i.ClearField('quantization_param')
            break

    with open(current_quantized_file, 'w') as f:
        f.write(str(net))


def tuning_quantized_topology(base_top1_accuracy, quantized_file, caffe_bin, model_weights, iterations,
                              enable_floating_point, toleration, model_type):
    if enable_floating_point == 0:
        print 'Updating quantization parameter...'
        transform_convolutions(quantized_file)
    current_top1_accuracy = get_the_accuracy(caffe_bin, quantized_file, model_weights, iterations, model_type)
    while abs(current_top1_accuracy - base_top1_accuracy) >= toleration:
        print 'Tuning... '
        print abs(current_top1_accuracy - base_top1_accuracy)
        remove_top_quantized_parameter(quantized_file)
        current_top1_accuracy = get_the_accuracy(caffe_bin, quantized_prototxt, model_weights, iterations, model_type)


if __name__ == '__main__':
    usage_string = 'Usage: 1.Build the caffe\n ' \
                   '2.cd /path/to/caffe/scripts\n ' \
                   '3.python calibrator.py ' \
                   ' -r /path/to/caffe/build ' \
                   ' -w pre-trained-fp32 weights ' \
                   ' -m typology ' \
                   ' -i iterations ' \
                   ' -t resnet|inceptionv3|ssd\n '

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help=usage_string)

    parser.add_argument('-i', '--iterations', action='store', dest='iterations', default=10,
                        help='iterations')

    parser.add_argument('-w', '--weights', action='store', dest='weights', default='',
                        help='pre-trained-fp32-weights')

    parser.add_argument('-m', '--model', action='store', dest='model', default='',
                        help='model')

    parser.add_argument('-l', '--accuracy_loss', action='store', dest='loss', default=0.01,
                        help='accuracy-loss')

    parser.add_argument('-t', '--type', action='store', dest='input_model_type', default='',
                        help='model type')

    parser.add_argument('-r', '--root', action='store', dest='root', default='',
                        help='caffe build path')
    
    params = parser.parse_args()

    try:
        iterations = int(params.iterations)
    except:
        print 'Set the iterations to the default value 1000'
        iterations = 1000

    try:
        toleration = float(params.loss)
        if toleration >= 1 or toleration < 0:
            toleration = 0.01
    except:
        print 'Set the toleration to 1%.'
        toleration = 0.01

    model = os.path.abspath(params.model)
    weights = os.path.abspath(params.weights)
    sample = os.path.abspath(params.root + 'tools/sample')
    caffe_bin = os.path.abspath(params.root + 'tools/caffe')
    setup_env()

    if params.input_model_type == 'resnet':
        model_type = 1
    elif params.input_model_type == 'inceptionv3':
        model_type = 2
    elif params.input_model_type == 'ssd':
        model_type = 3
    else:
        print 'Invalid model type!'
        sys.exit(-1)

    if check_existence(model) is False or check_existence(weights) is False or check_existence(sample) is False or \
            check_existence(caffe_bin) is False:
        print 'Please check model/weights/sample existence.'
        sys.exit(-1)

    sys.path.insert(0, params.root + '../python')
    quantized_prototxt = model.rsplit('.')[0] + '_quantized.prototxt'
    quantized_weights = weights.rsplit('.')[0] + '_quantized.caffemodel'
    enable_floating_point = 0
    print 'Sampling...'
    generate_sample(sample, model, weights,
                    quantized_prototxt, model_type, 10, 100 * toleration, enable_floating_point)

    print 'Sampling done'
    print 'Generating the FP32 accuracy...'
    top_1 = get_the_accuracy(caffe_bin, model, weights, iterations, model_type)
    print 'FP32 accuracy is: {}'.format(top_1)

    tuning_quantized_topology(top_1, quantized_prototxt, caffe_bin, weights, iterations, enable_floating_point,
                              toleration, model_type)

    print 'Updated prototxt {} is generated.'.format(quantized_prototxt)
