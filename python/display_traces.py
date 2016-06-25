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
    
    fig4 = plt.figure()
    plot_test_score(trace_digest, fig=fig4)
    plot_train_loss(trace_digest, fig=fig4, smoothing_window=10)
    fig4.suptitle("test score")
    
    plt.show()
    
def plot_activation_trace(trace_digest, fig=None, blobs=[]):
    '''
    Plot the activations from a proto buffer
    @param trace_digest protobuffer created by parsing the trace file
    @param fig Optional figure that the traces should be plotted to
    @param blobs If set, plot only the activation traces of these blobs
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
    
def plot_weight_trace(trace_digest, fig=None, return_traces=False, traces=set([])):
    return plot_trace(trace_digest.weight_trace_point, fig, return_traces, traces)
    
def plot_diff_trace(trace_digest, fig=None, return_traces=False, traces=set([])):
    '''
    Plot the trace of the diffs through training time
    @param trace_digest protobuffer created by parsing the trace file
    @param fig Optional figure that the traces should be plotted to
    @param return_traces If set to true, values of the traces are returned
    @param traces If set, plot only traces of layers in this set, otherwise everything
    '''
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

    if len(traces) == 0:
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

def plot_test_loss(trace_digest, fig=None, target=None, return_traces=False):
    x_test,y_test = {},{}

    for tp in trace_digest.test_loss_trace_point:
        if target is None or tp.test_net_id in target:
            if tp.test_net_id in x_test:
                x_test[tp.test_net_id].append(tp.iter)
                y_test[tp.test_net_id].append(tp.test_loss)
            else:
                x_test[tp.test_net_id] = [tp.iter]
                y_test[tp.test_net_id] = [tp.test_loss]
    
    for test_name in x_test.keys():
        if fig == None:
            plt.plot(x_test[test_name],y_test[test_name], label='test net id: ' + str(test_name))
            plt.legend()
        else:
            ax = fig.add_subplot(111)
            ax.plot(x_test[test_name],y_test[test_name], label='test net id: ' + str(test_name))
            ax.legend()
    if return_traces:
        return x_test,y_test
        
def plot_test_score(trace_digest, fig=None, target=None, return_traces=False):
    x_test,y_test = {},{}

    for tp in trace_digest.test_score_trace_point:
        if target is None or tp.test_net_id in target:
            id_string = 'test net id, score name: '
            id_string += str(tp.test_net_id) + ' ' + str(tp.score_name)
            if id_string in x_test:
                x_test[id_string].append(tp.iter)
                y_test[id_string].append(tp.mean_score)
            else:
                x_test[id_string] = [tp.iter]
                y_test[id_string] = [tp.mean_score]
    if fig:
        ax = fig.add_subplot(111)

    for test_name in x_test.keys():
        if fig == None:
            plt.plot(x_test[test_name],y_test[test_name], label=test_name)
            plt.legend()
        else:
            ax.plot(x_test[test_name],y_test[test_name], label=test_name)
            ax.legend()
    if return_traces:
        return x_test,y_test

def plot_train_loss(trace_digest, fig=None, return_traces=False, smoothing_window=1):
    assert smoothing_window >= 0
    x_test,y_test = [], []
    moving_window = np.zeros(smoothing_window, dtype=float)
    for i, tp in enumerate(trace_digest.train_trace_point):
        x_test.append(tp.iter)
        moving_window[i % smoothing_window] = tp.train_loss
        if i >= smoothing_window:
            y_test.append(np.mean(moving_window))
        else:
            y_test.append(np.sum(moving_window) / (i+1))
    if fig:
        ax = fig.add_subplot(111)

    if fig == None:
        plt.plot(x_test, y_test, label='training loss')
        plt.legend()
    else:
        ax.plot(x_test,y_test, label='training loss')
        ax.legend()
    if return_traces:
        return x_test,y_test

if __name__ == '__main__':
    main(sys.argv)
