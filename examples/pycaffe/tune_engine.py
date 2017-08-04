import os
import sys
import argparse
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import utils

def selectOptimalEngine(layers):
    optimal_layer = None
    min_time = sys.float_info.max
    for layer in layers:
        if layer[2] < min_time:
            min_time = layer[2]
            optimal_layer = layer

    return optimal_layer

def tuneEngine(logs, model):
    if len(logs) <= 1:
        print "[ERROR] Please specify two or more log files"
        exit(1)

    for log in logs:
        if not os.path.exists(log):
            print "[ERROR] Please specify valid log file:", log
            exit(1)

    layer_map = {}
    net = None
    for log in logs:
        log_name = os.path.basename(log)
        (model_str, time_lines) = utils.parseLog(log)
        (net, layer_model_map) = utils.parseModelStr(model_str)
        layer_time_map = utils.parseTimeLines(time_lines)
        for k, v in layer_model_map.items():
            if k not in layer_map.keys():
                layer_map[k] = [(v[0], v[1], layer_time_map[k], v[2])]
            else:
                layer_map_v = layer_map[k]
                layer_map_v.append((v[0], v[1], layer_time_map[k], v[2]))
                layer_map[k] = layer_map_v

    optimal_layer_map = {}
    for k, v in layer_map.items():
        optimal_layer = selectOptimalEngine(v)
        assert(optimal_layer != None)
        optimal_layer_map[optimal_layer[0]] = optimal_layer[3]
        
    genModel(net, model, optimal_layer_map)

def genModel(net, model, optimal_layer_map):
    net_str = ""
    net_str += "name: \"" + net.name + "\"\n"
    for index in range(0, len(net.layer)):
        net_str += "layer {\n"
        l = net.layer[index]
        if l.type.endswith("Data"):
            net_str += str(l) + "\n}\n"
            continue
        l = optimal_layer_map[index]
        net_str += str(l) + "\n}\n"
    with open(model, 'w') as f:
        net = caffe_pb2.NetParameter()
        txtf.Merge(net_str, net)
        f.write(str(net))
        print "[INFO] Complete model engine tuning:", model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--logs', nargs='+', help='require the caffe time logs', required=True)

    parser.add_argument('-o', '--output', action='store', dest='output', default="",
                        help='require the model output')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    params = parser.parse_args()
    if params.output == "":
        print "Please specify the output for tuned model with -o"
        sys.exit(1)

    tuneEngine(params.logs, params.output)
