import os
import sys
import numpy
import copy
#os.environ['GLOG_minloglevel'] = '2'
caffe_root="../"
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import argparse
import subprocess

def check_existence(path):
    try:
        return os.path.exists(path)
    except Exception as e:
        raise("Failed to check {} existence due to {}".format(path, str(e)))


def setup_env():
    os.chdir(os.path.dirname(os.path.abspath(params.root)))
    caffe.set_mode_cpu()


def read_prototxt(prototxt):
    try:
        if not check_existence(prototxt):
            return None;
       
        net = caffe_pb2.NetParameter()
       
        with open(prototxt) as f:
            txtf.Merge(f.read(), net)
        
        return net
    
    except Exception as e:
        raise("Failed to read {} due to {}".format(prototxt, str(e)))


def get_bottom_layers(layerName, net, start):
    bottomLayers = []
    for index, value in enumerate(net.layer[start:]):
        for sub_index, sub_value in enumerate(value.bottom):
            if(sub_value == layerName):
                bottomLayers.append((index, value.name, value.type))

    return bottomLayers


def get_top_layers(l, net, end):
    topLayers = []
    for layerIndex in range(0, end):
        reverseLayerIndex = end - layerIndex - 1
        for blobIndex in range(0, len(net.layer[reverseLayerIndex].top)):
            if (net.layer[reverseLayerIndex].top[blobIndex] in l.bottom):
                topLayers.append((reverseLayerIndex, net.layer[reverseLayerIndex].name, net.layer[reverseLayerIndex].type))      
    return topLayers


def GetAllTopLayers(l, net, end, skipLayers, interestingLayers):
    allTopLayers = []
    topLayers = get_top_layers(l, net, end)
    while True:
        if len(topLayers) == 0:
            break
    
        processedLayers = topLayers # sync topLayers change
        for (li, ln, lt) in processedLayers:
            if lt in skipLayers:
                topLayers.remove((li, ln, lt))
                continue
            if lt in interestingLayers:
                lp = (li, ln, lt)
                topLayers.remove(lp)
                if lp not in allTopLayers:
                    allTopLayers.append(lp)
                continue
     
            newTopLayers = get_top_layers(net.layer[li], net, li)
            topLayers.remove((li, ln, lt))
            topLayers.extend(newTopLayers)

    return allTopLayers

def GetAllBottomLayers(layerName, net, start, skipLayers, interestingLayers):
    allBottomLayers = []
    bottomLayers = get_bottom_layers(layerName, net, start)
    while True:
        if len(bottomLayers) == 0:
            break
    
        processedLayers = bottomLayers # sync bottomLayers change
        for (li, ln, lt) in processedLayers:
            if lt in skipLayers:
                bottomLayers.remove((li, ln, lt))
                continue
            if lt in interestingLayers:
                lp = (li, ln, lt)
                bottomLayers.remove(lp)
                if lp not in allBottomLayers:
                    allBottomLayers.append(lp)
                continue
     
            newBottomLayers = get_bottom_layers(ln, net, li + 1)
            bottomLayers.remove((li, ln, lt))
            bottomLayers.extend(newBottomLayers)

    return allBottomLayers


def transform_convolutions(model_path):
    net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, net)

    newNet = copy.deepcopy(net)
    
    convolutionLayers = [(value, index) for index, value in enumerate(net.layer) if value.type == 'Convolution']

    interestingLayers = ['ReLU']
    skipLayers = ['Convolution', 'Eltwise', 'Concat']

    u8_max = 255
    s8_max = 127

    for (l, index) in convolutionLayers:
        outputWithRelu = GetAllBottomLayers(l.name, net, index + 1, skipLayers, interestingLayers)
        inputWithRelu = GetAllTopLayers(l, net, index, skipLayers, interestingLayers)
        # print "Processing", l.type, l.name

        output_type = 'u8' if outputWithRelu else 's8'
        input_type = 'u8' if inputWithRelu else 's8'

        for si in range(0, len(newNet.layer[index].quantization_param.scale_out)):
            if len(outputWithRelu) > 0: #u8
                newNet.layer[index].quantization_param.scale_out[si] = round(u8_max / newNet.layer[index].quantization_param.scale_out[si], 2)
            else: #s8
                newNet.layer[index].quantization_param.scale_out[si] = round(s8_max / newNet.layer[index].quantization_param.scale_out[si], 2)

        for si in range(0, len(newNet.layer[index].quantization_param.scale_in)):
            if len(inputWithRelu) > 0: #u8
                newNet.layer[index].quantization_param.scale_in[si] = round(u8_max / newNet.layer[index].quantization_param.scale_in[si], 2)
            else: #s8
                newNet.layer[index].quantization_param.scale_in[si] = round(s8_max / newNet.layer[index].quantization_param.scale_in[si], 2)

        for si in range(0, len(newNet.layer[index].quantization_param.scale_params)):
            newNet.layer[index].quantization_param.scale_params[si] = round(s8_max / newNet.layer[index].quantization_param.scale_params[si], 2)

    with open(model_path, 'w') as f:
        f.write(str(newNet))


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
            return  float(top_1.split('=')[-1].strip())
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


def update_caffemodel(quantized_prototxt, caffemodel_file, target_weights):
    caffenet = caffe.Net(quantized_prototxt,
                    caffe.TEST)
    caffenet.copy_from(caffemodel_file)
    caffenet.save(target_weights)


def tuning_quantized_topology(base_top1_accuracy, quantized_file, caffe_bin,  model_weights, iterations, enable_floating_point, toleration, model_type):
    if enable_floating_point == 0:
        print 'Updating quantization parameter...'
        transform_convolutions(quantized_file)
    current_top1_accuracy = get_the_accuracy(caffe_bin, quantized_file, model_weights, iterations, model_type)
    while abs(current_top1_accuracy - base_top1_accuracy) >= toleration:
        print 'Tuning... '
        print  abs(current_top1_accuracy - base_top1_accuracy)
        remove_top_quantized_parameter(quantized_file)
        current_top1_accuracy  = get_the_accuracy(caffe_bin, quantized_prototxt, model_weights, iterations, model_type)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--iterations', action='store', dest='iterations', default=10,
                        help='iterations')

    parser.add_argument('-f', '--floatingpoint', action='store', dest='floatingpoint', default=0,
                        help='floatingpoint')

    parser.add_argument('-w', '--weights', action='store', dest='weights', default='',
                        help='weights')

    parser.add_argument('-m', '--model', action='store', dest='model', default='',
                        help='model')

    parser.add_argument('-l', '--loss', action='store', dest='loss', default=0.01,
                        help='toleration')

    parser.add_argument('-t', '--type', action='store', dest='input_model_type', default='',
                        help='model type')

    parser.add_argument('-r', '--root', action='store', dest='root', default='',
                        help='caffe build path')
    
    parser.add_argument('-k', '--keep_model', action='store', dest='keep_model', default=0,
                        help='keep_model')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    params = parser.parse_args()


    try:
        iterations = int(params.iterations)
    except:
        print 'Set the iterations to the default value 1000'
        iterations = 1000

    try:
        enable_floating_point = 0 if int(params.floatingpoint) == 0 else 1
    except:
        print 'Enable the floating point.'
        enable_floating_point = 0
    try:
        toleration = float(params.loss)
        if toleration >= 1 or toleration < 0:
            toleration = 0.01
    except:
        print 'Set the toleration to 1%.'
        toleration = 0.01
    
    try:
        keep_model = int(params.keep_model)
    except:
        keep_model = 0

    model = os.path.abspath(params.model)
    weights =  os.path.abspath(params.weights)
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


    if check_existence(model) is False or check_existence(weights) is False or check_existence(sample) is False or check_existence(caffe_bin) is False:
        print 'Please check model/weights/sample existence.'
        sys.exit(-1)
    
    sys.path.insert(0, params.root + '../python')
    quantized_prototxt = model.rsplit('.')[0] + '_quantized.prototxt'
    quantized_weights = weights.rsplit('.')[0] + '_quantized.caffemodel'
    print 'Sampling...'
    generate_sample(sample, model, weights,
                    quantized_prototxt, model_type, 10, 100*toleration, enable_floating_point)

    print 'Sampling done'
    print 'Generating the FP32 accuracy...'
    top_1  = get_the_accuracy(caffe_bin, model, weights, iterations, model_type)
    print 'FP32 accuracy is: {}'.format(top_1)
    
    tuning_quantized_topology(top_1, quantized_prototxt, caffe_bin,  weights, iterations,
                             enable_floating_point, toleration, model_type)

    update_caffemodel(quantized_prototxt, weights, quantized_weights)

    if keep_model:
        print 'Updated prototxt {} is generated.'.format(quantized_prototxt)
    else:
        try:
            os.remove(quantized_prototxt)
        except Exception as e:
            print 'Failed to remove {0} due to {1}'.format(quantized_prototxt, str(e))
    
    print 'Updated weights {} is generated.'.format(quantized_weights)


