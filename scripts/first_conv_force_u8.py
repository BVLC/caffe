import numpy as np
import os
import cv2
import sys
pycaffe = os.path.split(os.path.realpath(__file__))[0] + '/../python'
sys.path.insert(0, pycaffe)
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


def first_conv_u8_input(quantized_prototxt, weights, clx = False, force_u8_prototxt = '', new_weights_path = ''):

    if new_weights_path == '':
        new_weights_path = weights.split(".caffemodel")[0] + "_new.caffemodel"
    if force_u8_prototxt == '':
        force_u8_prototxt = quantized_prototxt.split('.prototxt')[0] + "_force_u8.prototxt"

    if os.path.isfile(quantized_prototxt):
        print ('caffeNet found.')
    else:
        print ('caffenet not found')

    net = caffe.Net(quantized_prototxt, weights, caffe.TEST)

    proto_net = caffe_pb2.NetParameter()
    with open(quantized_prototxt) as f:
        s = f.read()
        txtf.Merge(s, proto_net)

    for index, layer in enumerate(proto_net.layer):
        if layer.type == "Data":
            data_layer = layer
        if layer.type == "Convolution":
            first_conv_layer = layer
            break

    first_conv_weight = net.params[first_conv_layer.name][0].data
    first_conv_bias = net.params[first_conv_layer.name][1].data

    scale_in = first_conv_layer.quantization_param.scale_in[0]
    bias_scale = 128/scale_in
    new_bias = np.zeros(first_conv_bias.shape)
    for i in range(len(first_conv_bias)):
        new_bias[i] = first_conv_bias[i] - bias_scale * np.sum(first_conv_weight[i])

    net.params[first_conv_layer.name][1].data[...] = new_bias

    net.save(new_weights_path)

    first_conv_layer.quantization_param.is_negative_input = False
    old_pad = first_conv_layer.convolution_param.pad[0]
    first_conv_layer.convolution_param.pad[0] = 0L
    first_conv_layer.quantization_param.force_u8_input = True
    if not clx:
        for i in range(len(first_conv_layer.quantization_param.scale_params)):
            first_conv_layer.quantization_param.scale_params[i] = float(first_conv_layer.quantization_param.scale_params[i]) * 0.5

    data_layer.transform_param.pad = long(old_pad)
    mean_value = np.array(data_layer.transform_param.mean_value).astype(float)
    scale = data_layer.transform_param.scale
    new_scale = scale_in * scale
    new_mean_value = mean_value - 128/new_scale
    data_layer.transform_param.mean_value[0] = new_mean_value[0]
    data_layer.transform_param.mean_value[1] = new_mean_value[1]
    data_layer.transform_param.mean_value[2] = new_mean_value[2]
    
    data_layer.transform_param.scale = new_scale

    with open(force_u8_prototxt, 'w') as f:
        f.write(str(proto_net))

    print("Please use new prototxt in {0}, caffemodel in {1}".format(force_u8_prototxt, new_weights_path))

if __name__=='__main__':
    if len(sys.argv) < 2:
        print "Usgae: python first_conv_force_u8.py $prototxt $weights"
        sys.exit(0)
    first_conv_u8_input(sys.argv[1], sys.argv[2])
