import os
import datetime
import copy
import argparse

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import caffe

def isWinogradApplicable(ic, oc, stride, kernel_size):
    if ic % 16 != 0:
        return False
    if oc % 16 != 0:
        return False
    if stride != 1:
        return False
    if kernel_size != 3:
        return False

    return True

def genHybridModel(net, winogradLayers, modelName):
    newNet = copy.deepcopy(net)
    newNetName = modelName.split(".")[0] + "_hybrid.prototxt"
    for layer in winogradLayers:
        newNet.layer[layer].convolution_param.conv_algorithm = "winograd"
    with open(newNetName, 'w') as f:
       f.write(str(newNet))
       print "[INFO] Complete model tuning with Winograd:", newNetName

def tuneModelDefinition(model):
    net = caffe_pb2.NetParameter()
    with open(model) as f:
        s = f.read()
        txtf.Merge(s, net)

    net.name = 'Tuned model of ' + net.name
    output_layer_map = {} 
    for index in range(0, len(net.layer)):
        l = net.layer[index]
        if l.type == ("Convolution"):
            stride = 0
            kernel_size = 0
            if len(l.convolution_param.stride) == 0:
                stride = 1
            else:
                stride = l.convolution_param.stride[0]
            kernel_size = l.convolution_param.kernel_size[0]
            ic = 0
            if l.bottom[0] in output_layer_map.keys():
                ic = output_layer_map[l.bottom[0]][4]
            oc = l.convolution_param.num_output
            output_layer_map[l.name] = (index, stride, kernel_size, ic, oc, True)
        elif l.type == ("InnerProduct"):
            oc = l.inner_product_param.num_output
            ic = 0
            if l.bottom[0] in output_layer_map.keys():
                ic = output_layer_map[l.bottom[0]][4]
            output_layer_map[l.name] = (index, 0, 0, ic, oc, False)
        elif l.type.endswith("Data") or l.type.endswith("Input"):
            # TODO: correct the output
            #    dynamic_net = caffe.Net(model, caffe.TEST)
            #    for k, v in dynamic_net.blobs.items():
            #        dynamic_net_map[k] = v.data.shape
            ic = oc = 3
            output_layer_map[l.name] = (index, 0, 0, ic, oc, False)
        else:
            ic = 0
            if l.bottom[0] in output_layer_map.keys():
                ic = output_layer_map[l.bottom[0]][4]
            oc = ic
            output_layer_map[l.name] = (index, 0, 0, ic, oc, False)

    winograd_convolutions = []
    for k,v in output_layer_map.items():
        if v[5] and isWinogradApplicable(v[3], v[4], v[1], v[2]):
            winograd_convolutions.append(v[0])

    if len(winograd_convolutions) > 0:
        genHybridModel(net, winograd_convolutions, model)
    else:
        print "[INFO] No need to tune model with Winograd:", model
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', action='store', dest='model', default="",
                        help='require the model definition (prototxt)')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    params = parser.parse_args()

    model = params.model
    if not os.path.exists(params.model):
        print "[ERROR] Please specify the model definition file with -m"
        exit(1)

    tuneModelDefinition(model)
