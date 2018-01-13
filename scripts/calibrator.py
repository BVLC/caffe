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
        print ("Failed to read {} due to {}".format(prototxt, e))


def get_bottom_layers(top_name, net, start):
    bottom_layers = []
    for index, value in enumerate(net.layer[start:]):
        for sub_index, sub_value in enumerate(value.bottom):
            if sub_value == top_name:
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


def get_all_bottom_layers(top_name, net, start, skip_layers, interesting_layers):
    all_bottom_layers = []
    bottom_layers = get_bottom_layers(top_name, net, start)
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


def get_fusion_conv_names(compiled_model):
    compiled_net = caffe_pb2.NetParameter()
    with open(compiled_model) as f:
        s = f.read()
        txtf.Merge(s, compiled_net)
    return [(layer.name, layer.bottom[1]) for _, layer in enumerate(compiled_net.layer)
            if layer.type == 'Convolution' and len(layer.bottom) > 1]


def filter_fusion_layers(net, fusion_layer, conv_layer):
    if not fusion_layer or not conv_layer:
        return []
    interesting_layers = ['ReLU']
    skip_layers = ['Convolution', 'Eltwise', 'Concat']
    output_with_relu_layer = [(l.name, net.layer[index].top[0]) for l, index in conv_layer
                              if len(get_all_bottom_layers(net.layer[index].top[0], net, index + 1,
                                                           skip_layers, interesting_layers)) == 0]
    output_without_dict = {v: k for (k, v) in output_with_relu_layer}
    for layer_name, top_name in fusion_layer:
        if top_name in output_without_dict.keys():
            del output_without_dict[top_name]

    return output_without_dict.values()


def check_relu_existence(net, start, end, exclude_layer):
    if net.layer[start].type == 'Convolution' and net.layer[start].name in exclude_layer:
        return False

    for i in net.layer[start + 1: end]:
        if i.type == 'ReLU':
            return True
    return False


def transform_convolutions(model_path, compiled_model_path):
    net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, net)

    fusion_layer = get_fusion_conv_names(compiled_model_path)
    new_net = copy.deepcopy(net)

    convolution_layers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'Convolution']

    interesting_layers = ['ReLU']
    skip_layers = ['Convolution', 'Eltwise', 'Concat']

    u8_max = 255
    s8_max = 127
    u8_layers = filter_fusion_layers(net, fusion_layer, convolution_layers)
    for (l, index) in convolution_layers:
        outputwith_relu = get_all_bottom_layers(net.layer[index].top[0], net, index + 1, skip_layers,
                                                interesting_layers)
        conv_relu_flag = check_relu_existence(net, index,
                                              convolution_layers[convolution_layers.index((l, index)) + 1][1]
                                              if (l, index) != convolution_layers[-1]
                                              else len(net.layer), [i[0] for i in fusion_layer])
        inputwith_relu = get_all_top_layers(l, net, index, skip_layers, interesting_layers)

        for si in range(0, len(new_net.layer[index].quantization_param.scale_out)):
            if len(outputwith_relu) > 0 or l.name in u8_layers or conv_relu_flag:  # u8
                new_net.layer[index].quantization_param.scale_out[si] = round(u8_max / new_net.layer[index].
                                                                              quantization_param.scale_out[si], 2)
            else:  # s8
                new_net.layer[index].quantization_param.scale_out[si] = round(s8_max / new_net.layer[index].
                                                                              quantization_param.scale_out[si], 2)

        for si in range(0, len(new_net.layer[index].quantization_param.scale_in)):
            if len(inputwith_relu) > 0 or l.type == 'Convolution':  # u8
                new_net.layer[index].quantization_param.scale_in[si] = round(u8_max / new_net.layer[index].
                                                                             quantization_param.scale_in[si], 2)
            else:
                new_net.layer[index].ClearField('quantization_param')
                continue

        for si in range(0, len(new_net.layer[index].quantization_param.scale_params)):
            new_net.layer[index].quantization_param.scale_params[si] = round(s8_max / new_net.layer[index].
                                                                             quantization_param.scale_params[si], 2)

    with open(model_path, 'w') as f:
        f.write(str(new_net))


def generate_sample(sample_path, input_model, weights,
                    quantized_model, detection, iterations=1, error_margin=1, power=0):
    cmd = '{0} quantize -model {1} -weights {2} -model_quantized {3} -iterations {4} ' \
          '-trimming_mode dynamic_fixed_point -error_margin {5} -power {6}'.format(sample_path, input_model, weights,
                                                                                   quantized_model, iterations,
                                                                                   error_margin, power)
    if detection:
        cmd += ' --detection=1'

    os.system(cmd)


def get_compiled_net(caffe_bin, model_def, model_weights, detection):
    output_log_name = '.compiled_net.txt'

    cmd = '{} test -model {} -weights {} -iterations 1'.format(caffe_bin, model_def, model_weights)
    if detection:
        cmd += ' -detection'
    cmd += ' 2>&1 > {}'.format(output_log_name)

    os.system(cmd)
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
                return float(i.split('=')[-1].strip())
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


def tuning_quantized_topology(base_top1_accuracy, prototxt, caffe_bin, model_weights, iterations,
                              is_floating_point, accuracy_loss, detection, blob_name):
    if is_floating_point == 0:
        print 'Updating quantization parameter...'

        transform_convolutions(prototxt, get_compiled_net(caffe_bin, prototxt, model_weights, detection))

    current_top1_accuracy = get_the_accuracy(caffe_bin, prototxt, model_weights, iterations, detection, blob_name)

    while abs(current_top1_accuracy - base_top1_accuracy) >= accuracy_loss:
        print 'Tuning... '
        print abs(current_top1_accuracy - base_top1_accuracy)
        remove_top_quantized_parameter(prototxt)
        current_top1_accuracy = get_the_accuracy(caffe_bin, prototxt, model_weights, iterations, detection, blob_name)


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


if __name__ == '__main__':
    usage_string = 'Usage: 1.Build the caffe\n ' \
                   '2.cd /path/to/caffe/scripts\n ' \
                   '3.python calibrator.py ' \
                   ' -r /path/to/caffe/build ' \
                   ' -w pre-trained-fp32 weights ' \
                   ' -m typology ' \
                   ' -i iterations ' \
                   ' -l acceptable accuracy loss value, the default value is 0.01(stands for 1%)' \
                   ' -d 1(0 means classification while 1 means detection, the default value is 0' \
                   ' -n blob name which means accuracy.\n '

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
                             'default value is 0.01(1%).')

    parser.add_argument('-d', '--detection', action='store', dest='is_detection', default=0,
                        help='0 for classification while 1 for detection, default value is 0.')

    parser.add_argument('-r', '--root', action='store', dest='root', default='',
                        help='caffe build path')

    parser.add_argument('-n', '--blob_name', action='store', dest='blob_name', default='',
                        help='top blob name which stands for accuracy')
    params = parser.parse_args()

    try:
        user_input_iterations = int(params.iterations)
    except:
        print 'Set the iterations to the default value 1000'
        user_input_iterations = 1000

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

    model = os.path.abspath(params.model)
    user_input_weights = os.path.abspath(params.weights)
    sample = os.path.abspath(params.root + 'tools/sample')
    caffe_bin_path = os.path.abspath(params.root + 'tools/caffe')
    setup_env()

    if not check_existence(model) or not check_existence(user_input_weights) or not check_existence(sample) \
            or not check_existence(caffe_bin_path):
        print 'Please check model/weights/sample existence.'
        sys.exit(-1)

    target_blob_name = params.blob_name
    if not target_blob_name or not check_blob_name_existence(model, target_blob_name):
        print 'Please specify valid blob name and rerun the script.'
        sys.exit(-1)

    sys.path.insert(0, params.root + '../python')
    quantized_prototxt = model.rsplit('.')[0] + '_quantized.prototxt'
    quantized_weights = user_input_weights.rsplit('.')[0] + '_quantized.caffemodel'
    enable_floating_point = 0
    print 'Sampling...'
    generate_sample(sample, model, user_input_weights,
                    quantized_prototxt, detection_flag, 10, 100 * toleration, enable_floating_point)
    print 'Sampling done'
    print 'Generating the FP32 accuracy...'
    top_1 = get_the_accuracy(caffe_bin_path, model, user_input_weights, user_input_iterations, detection_flag,
                             target_blob_name)
    print 'FP32 accuracy is: {}'.format(top_1)
    tuning_quantized_topology(top_1, quantized_prototxt, caffe_bin_path, user_input_weights, user_input_iterations,
                              enable_floating_point, toleration, detection_flag, target_blob_name)

    print 'Updated prototxt {} is generated.'.format(quantized_prototxt)
