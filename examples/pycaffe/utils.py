import os
import sys
sys.path.insert(0,"python")
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

def readFile(filePath):
    lines = []
    file = open(filePath, 'r')
    for line in file.readlines():
        lines.append(line)
    file.close()

    return lines

def writeFile(filePath, lines):
    file = open(filePath, 'w+')
    file.write(lines)
    file.close()

def parseLog(log):
    lines = readFile(log)
    model_start = False
    time_start = False
    model_lines = []
    time_lines = []
    for line in lines:
        trim_line = line.strip()
        if trim_line.endswith("Initializing net from parameters:"):
            model_start = True
            continue
        if model_start:
            if trim_line.find("Creating layer") <> -1:
                model_start = False
                continue
            model_lines.append(line)

        if trim_line.endswith("Average time per layer:"):
            time_start = True
            continue
        if time_start:
            if trim_line.find("Average Forward pass") <> -1:
                time_start = False
                break
            time_lines.append(line)

    model_lines = model_lines[1:]
    model_str = ""
    for line in model_lines:
        model_str = model_str + line

    return (model_str, time_lines)

def parseTimeLines(timeLines):
    layer_map = {}
    for line in timeLines:
        trim_line = line.strip()
        items = trim_line.split("\t")
        layer_items = items[0].split(" ")
        layer_name = layer_items[-1]
        time_items = items[1].split(" ")
        if layer_name not in layer_map.keys():
            layer_map[layer_name] = (float)(time_items[1])
        else:
            layer_map[layer_name] = layer_map[layer_name] + (float)(time_items[1])

    return layer_map

def parseModelStr(modelStr):
    net = caffe_pb2.NetParameter()
    txtf.Merge(modelStr, net)
    layer_model_map = {}
    global_engine = "CAFFE"
    if net.engine != "":
        global_engine = net.engine
    for index in range(0, len(net.layer)):
        engine = global_engine
        l = net.layer[index]
        if l.engine != "":
            engine = l.engine
        param_engine = -1
        if l.type == "Convolution" or l.type == "Deconvolution":
            if l.convolution_param.engine != "":
                param_engine = l.convolution_param.engine
        elif l.type == "BatchNorm":
            if l.batch_norm_param.engine != "":
                param_engine = l.batch_norm_param.engine
        elif l.type == "Concat":
            if l.concat_param.engine != "":
                param_engine = l.concat_param.engine
        elif l.type == "Eltwise":
            if l.eltwise_param.engine != "":
                param_engine = l.eltwise_param.engine
        elif l.type == "InnerProduct":
            if l.inner_product_param.engine != "":
                param_engine = l.inner_product_param.engine
        elif l.type == "LRN":
            if l.lrn_param.engine != "":
                param_engine = l.lrn_param.engine
        elif l.type == "Pooling":
            if l.pooling_param.engine != "":
                param_engine = l.pooling_param.engine
        elif l.type == "ReLU":
            if l.relu_param.engine != "":
                param_engine = l.relu_param.engine

        if param_engine == 0 or param_engine == 1:
            engine = "CAFFE"
        elif param_engine == 3:
            engine = "MKL2017"
        elif param_engine == 4:
            engine = "MKLDNN"
        layer_model_map[l.name] = (index, engine, l)

    return (net, layer_model_map)
