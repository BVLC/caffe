#!/usr/bin/env python
"""
    Code for visualizing caffe network conv layers.

"""

import matplotlib
matplotlib.use('Agg')

from caffe.convert import blobproto_to_array
from caffe.proto import caffe_pb2

import pylab as pl
import numpy as np

from math import sqrt, ceil
import sys
import os


class ConvLayerVisualizer(object):

    def __init__(self, file_name):
        self.load_model(file_name)
        self.extract_conv_layers()

    def load_model(self, file_name):
        """
            Load the snapshot into a NetParameter object.
        """
        net = caffe_pb2.NetParameter()
        data = open(file_name).read()
        net.ParseFromString(data)
        self.net = net

    def extract_conv_layers(self):
        """
            Extract all the convolutional layers from the network.
        """
        conv_layers = []
        for layer in self.net.layers:
            if layer.layer.type != "conv":
                continue
            conv_layers.append(layer.layer)
        self.conv_layers = conv_layers

    def visualize_conv_layers(self):
        """
            Visualize the conv layer kernels in a subplot each.

        """
        conv_layers = self.conv_layers

        f, axes = pl.subplots(len(conv_layers))
        f.suptitle(self.net.name)
        for idx, (conv_layer, ax) in enumerate(zip(conv_layers, axes)):
            W = blobproto_to_array(conv_layer.blobs[0])
            # only combine color channels for the first conv layer
            combine_chans = idx == 0
            plot_weights(W, title="Layer %d" % idx,
                         axis=ax, combine_chans=combine_chans)
        self._fig = f

    def visualize_conv_layer(self, conv_layer_idx=0):
        """
            Visualize a single conv layer of the network.

        """
        conv_layers = self.conv_layers

        W = blobproto_to_array(conv_layers[conv_layer_idx].blobs[0])
        # only combine color channels for the first conv layer
        combine_chans = conv_layer_idx == 0
        f, axis = pl.subplots()
        plot_weights(W, title="Layer %d" % conv_layer_idx,
                     axis=axis, combine_chans=combine_chans)
        self._fig = f

    def save_fig_to_file(self, file_name):
        self._fig.savefig(file_name)


def plot_weights(filters, title, axis=None, combine_chans=False):
    """
        Takes conv layer kernel numpy ndarray as an input.
        Plots all kernels in one big image.
   """
    filters = filters - filters.min()
    filters = filters / filters.max()

    if axis is None:
        f, axis = pl.subplots()

    make_filter_fig(filters,
                    filter_start=0,
                    axis=axis,
                    title=title,
                    num_filters=filters.shape[0],
                    combine_chans=combine_chans)


def make_filter_fig(filters,
                    filter_start,
                    axis,
                    title,
                    num_filters,
                    combine_chans):
    """
        Plot the given filters.

        filters:
            ndarray with dimensions:
                num_examples, num_channels, filter_size, filter_size

        Code adapted from:
          https://code.google.com/p/cuda-convnet/source/browse/trunk/shownet.py
    """
    FILTERS_PER_ROW = int(ceil(sqrt(filters.shape[0])))
    MAX_ROWS = FILTERS_PER_ROW
    MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
    num_colors = filters.shape[1]
    f_per_row = int(ceil(FILTERS_PER_ROW /
                    float(1 if combine_chans else num_colors)))
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))

    assert filters.shape[2] == filters.shape[3]
    filter_size = int(filters.shape[2])
    axis.set_title('%s %dx%d filters %d-%d' % (title, filter_size, filter_size,
                                               filter_start, filter_end-1),
                   horizontalalignment='center')
    num_filters = filter_end - filter_start
    if not combine_chans:
        bigpic = np.zeros((filter_size * filter_rows + filter_rows + 1,
                           filter_size*num_colors * f_per_row + f_per_row + 1),
                          dtype=np.single)
    else:
        bigpic = np.zeros((3, filter_size * filter_rows + filter_rows + 1,
                           filter_size * f_per_row + f_per_row + 1),
                          dtype=np.single)

    for m in xrange(filter_start, filter_end):
        filter = filters[m,:,:,:]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        if not combine_chans:
            for c in xrange(num_colors):
                filter_pic = filter[c,:].reshape((filter_size,filter_size))
                bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
        else:
            filter_pic = filter.reshape((3, filter_size, filter_size))
            bigpic[:,
                   1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                   1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
            
    axis.set_xticks([])
    axis.set_yticks([])
    if not combine_chans:
        axis.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    else:
        bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
        axis.imshow(bigpic, interpolation='nearest')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s input_net_proto_file output_image_file' % \
            os.path.basename(sys.argv[0])
    else:
        print "Loading %s" % sys.argv[1]
        visualizer = ConvLayerVisualizer(sys.argv[1])

        visualizer.visualize_conv_layer()

        print 'Exporting conv layers to %s' % sys.argv[2]
        visualizer.save_fig_to_file(sys.argv[2])
