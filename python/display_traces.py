#!/usr/bin/env python
"""
display_traces.py Command line callable to display trace created while training
"""

import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
import argparse

from caffe.proto import caffe_pb2 as proto # run 'make pycaffe' in the caffe directory 

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required argument: trace file
    parser.add_argument(
        "--trace_file",
        help=".caffetrace file to visualize"
    )
    args = parser.parse_args()
    with open(args.trace_file,'rb') as file_handle:
        proto_string = file_handle.read()
    trace_digest = proto.TraceDigest.FromString(proto_string)
    
    fig1 = plt.figure()
    plot_weight_trace(trace_digest, fig=fig1)
    fig1.suptitle("weight traces")
    
    fig2 = plt.figure()
    plot_diff_trace(trace_digest, fig=fig2)
    fig2.suptitle("diff traces")
    
    fig3 = plt.figure()
    plot_activation_trace(trace_digest, fig=fig3)
    fig3.suptitle("activation traces")
    
    plt.show()
    
def plot_activation_trace(trace_digest, fig=None, blobs=[]):
    '''
    Plot the activations from a proto buffer
    '''
    if fig==None:
        fig = plt.figure()
    else:
        plt.clf()
    count = 1
    for at in trace_digest.activation_trace:
        if blobs != []:
            num_blobs = len(blobs)
        else:
            num_blobs = len(trace_digest.activation_trace)
        if blobs != [] and at.blob_name not in blobs:
            continue
        num_activation_traces = len(at.activation_trace_point)
        ax = fig.add_subplot(num_blobs, 1, count)
        count += 1
        x = np.empty(num_activation_traces, dtype=float)

        num_traces = len(at.activation_trace_point[0].value)
        if num_traces == 0:
            break
        y = np.empty((num_activation_traces, num_traces), dtype=float)
        
        for j, atp in enumerate(at.activation_trace_point):
            x[j] = atp.iter
            for k, value in enumerate(atp.value):
                y[j,k] = value
        
        for j in range(num_traces):
            ax.plot(x, y[:,j])
        ax.set_title(at.blob_name)
    
def plot_weight_trace(trace_digest, fig=None, return_traces=False, traces={}):
    return plot_trace(trace_digest.weight_trace_point, fig, return_traces, traces)
    
def plot_diff_trace(trace_digest, fig=None, return_traces=False, traces={}):
    return plot_trace(trace_digest.diff_trace_point, fig, return_traces, traces)

def plot_trace(weight_trace_digest, fig, return_traces, traces):
    '''
    Plot weight or diff traces of blobs of layers saved in a proto buffer
    '''
    if fig==None:
        fig = plt.figure()
    else:
        plt.clf()
    
    # a lambda to append something to something other or nothing
    append = lambda x,y: y if x is None else np.append(x,y,axis=0)

    max_num_blobs = 0  
    #figure out how many blobs we need to print
    for wt in weight_trace_digest:
        if wt.param_id >= max_num_blobs:
            max_num_blobs = wt.param_id+1

    layers = {}
    x = {}
    for i,wt in enumerate(weight_trace_digest):
      
        if not layers.has_key(wt.layer_name):
            layers[wt.layer_name] = max_num_blobs * [ None ]
        
        layers[wt.layer_name][wt.param_id] = append( layers[wt.layer_name][wt.param_id], [np.array(wt.value)]);
        x[wt.iter]=None
    
    x=np.sort(x.keys());

    if traces=={}:
        traces = layers.keys()
    else:
        traces = set(layers.keys()).intersection(traces)

    i = 1
    rows = len(traces)
    for layer in np.sort(list(traces)):
        for j in range(max_num_blobs):
            if j < len(layers[layer]):
                ax = fig.add_subplot(rows,max_num_blobs,i)
                ax.plot(x,layers[layer][j])
                ax.set_title(layer+"_"+str(j))
            i+=1

    if return_traces:
        return layers, x

if __name__ == '__main__':
    main(sys.argv)