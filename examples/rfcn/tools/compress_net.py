#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compress a Fast R-CNN network using truncated SVD."""

import _init_paths
import caffe
import argparse
import numpy as np
import os, sys

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Compress a Fast R-CNN network')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the uncompressed network',
                        default=None, type=str)
    parser.add_argument('--def-svd', dest='prototxt_svd',
                        help='prototxt file defining the SVD compressed network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to compress',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def compress_weights(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.

    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain

    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(W, full_matrices=False)

    Ul = U[:, :l]
    sl = s[:l]
    Vl = V[:l, :]

    L = np.dot(np.diag(sl), Vl)
    return Ul, L

def main():
    args = parse_args()

    # prototxt = 'models/VGG16/test.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    # prototxt_svd = 'models/VGG16/svd/test_fc6_fc7.prototxt'
    # caffemodel = 'snapshots/vgg16_fast_rcnn_iter_40000.caffemodel'
    net_svd = caffe.Net(args.prototxt_svd, args.caffemodel, caffe.TEST)

    print('Uncompressed network {} : {}'.format(args.prototxt, args.caffemodel))
    print('Compressed network prototxt {}'.format(args.prototxt_svd))

    out = os.path.splitext(os.path.basename(args.caffemodel))[0] + '_svd'
    out_dir = os.path.dirname(args.caffemodel)

    # Compress fc6
    if net_svd.params.has_key('fc6_L'):
        l_fc6 = net_svd.params['fc6_L'][0].data.shape[0]
        print('  fc6_L bottleneck size: {}'.format(l_fc6))

        # uncompressed weights and biases
        W_fc6 = net.params['fc6'][0].data
        B_fc6 = net.params['fc6'][1].data

        print('  compressing fc6...')
        Ul_fc6, L_fc6 = compress_weights(W_fc6, l_fc6)

        assert(len(net_svd.params['fc6_L']) == 1)

        # install compressed matrix factors (and original biases)
        net_svd.params['fc6_L'][0].data[...] = L_fc6

        net_svd.params['fc6_U'][0].data[...] = Ul_fc6
        net_svd.params['fc6_U'][1].data[...] = B_fc6

        out += '_fc6_{}'.format(l_fc6)

    # Compress fc7
    if net_svd.params.has_key('fc7_L'):
        l_fc7 = net_svd.params['fc7_L'][0].data.shape[0]
        print '  fc7_L bottleneck size: {}'.format(l_fc7)

        W_fc7 = net.params['fc7'][0].data
        B_fc7 = net.params['fc7'][1].data

        print('  compressing fc7...')
        Ul_fc7, L_fc7 = compress_weights(W_fc7, l_fc7)

        assert(len(net_svd.params['fc7_L']) == 1)

        net_svd.params['fc7_L'][0].data[...] = L_fc7

        net_svd.params['fc7_U'][0].data[...] = Ul_fc7
        net_svd.params['fc7_U'][1].data[...] = B_fc7

        out += '_fc7_{}'.format(l_fc7)

    filename = '{}/{}.caffemodel'.format(out_dir, out)
    net_svd.save(filename)
    print 'Wrote svd model to: {:s}'.format(filename)

if __name__ == '__main__':
    main()
