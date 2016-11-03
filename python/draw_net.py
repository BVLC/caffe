#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Draw a graph of the net architecture.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
import caffe.draw
from caffe.proto import caffe_pb2


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_net_proto_file',
                        help='Input network prototxt file')
    parser.add_argument('output_image_file',
                        help='Output image file')
    parser.add_argument('--rankdir',
                        help=('One of TB (top-bottom, i.e., vertical), '
                              'RL (right-left, i.e., horizontal), or another '
                              'valid dot option; see '
                              'http://www.graphviz.org/doc/info/'
                              'attrs.html#k:rankdir'),
                        default='LR')
    parser.add_argument('--phase',
                        help=('Which network phase to draw: can be TRAIN, '
                              'TEST, or ALL.  If ALL, then all layers are drawn '
                              'regardless of phase.'),
                        default="ALL")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.input_net_proto_file).read(), net)
    print('Drawing net to %s' % args.output_image_file)
    phase=None;
    if args.phase == "TRAIN":
        phase = caffe.TRAIN
    elif args.phase == "TEST":
        phase = caffe.TEST
    elif args.phase != "ALL":
        raise ValueError("Unknown phase: " + args.phase)
    caffe.draw.draw_net_to_file(net, args.output_image_file, args.rankdir,
                                phase)


if __name__ == '__main__':
    main()
