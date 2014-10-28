#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
import os
from google.protobuf import text_format

import caffe, caffe.draw
from caffe.proto import caffe_pb2


def main(argv):
    if len(argv) != 3:
        print 'Usage: %s input_net_proto_file output_image_file' % \
                os.path.basename(sys.argv[0])
    else:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(sys.argv[1]).read(), net)
        print 'Drawing net to %s' % sys.argv[2]
        caffe.draw.draw_net_to_file(net, sys.argv[2])


if __name__ == '__main__':
    import sys
    main(sys.argv)
