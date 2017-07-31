import os
import sys
import argparse
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import copy
import utils

def genOptimalModel(net, mkldnn_time_map, mkldnn_winograd_time_map, optimal_model):
    for index in range(0, len(net.layer)):
        l = net.layer[index]
        if l.type == "Convolution":
            if mkldnn_winograd_time_map[l.name] <= mkldnn_time_map[l.name]:
                l.convolution_param.conv_algorithm = "winograd"
        
    with open(optimal_model, "w") as f:
        f.write(txtf.MessageToString(net, float_format=".17g"))

def tuneModelDefinition(model_path, iteration):
    working_dir = sys.path[0]
    caffe_path = os.path.join(working_dir, "..", "..", "build", "tools", "caffe")
    if not os.path.exists(caffe_path):
        print "Caffe binary does not exist; please build Caffe binary first."
        sys,exit(1)

    base_model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    winograd_model_name = base_model_name.split(".")[0] + "_winograd.prototxt"
    winograd_model_path = os.path.join(model_dir, winograd_model_name)

    base_net = caffe_pb2.NetParameter()
    with open(model_path) as f:
        s = f.read()
        txtf.Merge(s, base_net)

    net_copy = copy.deepcopy(base_net)
    for index in range(0, len(base_net.layer)):
        l = base_net.layer[index]
        if l.type == "Convolution":
            l.convolution_param.conv_algorithm = "winograd"

    with open(winograd_model_path, "w") as f:
        f.write(txtf.MessageToString(base_net, float_format=".17g"))

    mkldnn_log = "mkldnn.log"
    mkldnn_winograd_log = "mkldnn_winograd.log"
    mkldnn_log_path = os.path.join(model_dir, mkldnn_log)
    mkldnn_winograd_log_path = os.path.join(model_dir, mkldnn_winograd_log)
    
    mkldnn_command = caffe_path + " time -model " + model_path + " -engine MKLDNN -iterations " + str(iteration) + " >& " + mkldnn_log_path
    os.system(mkldnn_command)
    mkldnn_winograd_command = caffe_path + " time -model " + model_path + " -engine MKLDNN -iterations " + str(iteration) + " >& " + mkldnn_winograd_log_path
    os.system(mkldnn_winograd_command)

    logs = [mkldnn_log_path, mkldnn_winograd_log_path]

    (model_str, mkldnn_time_lines) = utils.parseLog(mkldnn_log_path)
    mkldnn_layer_time_map = utils.parseTimeLines(mkldnn_time_lines)
    (model_str, mkldnn_winograd_time_lines) = utils.parseLog(mkldnn_winograd_log_path)
    mkldnn_winograd_layer_time_map = utils.parseTimeLines(mkldnn_winograd_time_lines)

    hybrid_model_name = base_model_name.split(".")[0] + "_hybrid.prototxt"
    hybrid_model_path = os.path.join(model_dir, hybrid_model_name)
    genOptimalModel(net_copy, mkldnn_layer_time_map, mkldnn_winograd_layer_time_map, hybrid_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', action='store', dest='model', default="",
                        help='require the model definition (prototxt)')

    parser.add_argument('-i', '--iteration', action='store', dest='iterations', type=int, default=10,
                        help='require iterations number to run the model')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    params = parser.parse_args()

    model = params.model
    if not os.path.exists(params.model):
        print "[ERROR] Please specify the model definition file with -m"
        exit(1)

    tuneModelDefinition(params.model, params.iterations)
