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
import numpy as np
import unittest

import caffe

class TestBlobProtoToArray(unittest.TestCase):

    def test_old_format(self):
        data = np.zeros((10,10))
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.data.extend(list(data.flatten()))
        shape = (1,1,10,10)
        blob.num, blob.channels, blob.height, blob.width = shape

        arr = caffe.io.blobproto_to_array(blob)
        self.assertEqual(arr.shape, shape)

    def test_new_format(self):
        data = np.zeros((10,10))
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.data.extend(list(data.flatten()))
        blob.shape.dim.extend(list(data.shape))

        arr = caffe.io.blobproto_to_array(blob)
        self.assertEqual(arr.shape, data.shape)

    def test_no_shape(self):
        data = np.zeros((10,10))
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.data.extend(list(data.flatten()))

        with self.assertRaises(ValueError):
            caffe.io.blobproto_to_array(blob)

    def test_scalar(self):
        data = np.ones((1)) * 123
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.data.extend(list(data.flatten()))

        arr = caffe.io.blobproto_to_array(blob)
        self.assertEqual(arr, 123)


class TestArrayToDatum(unittest.TestCase):

    def test_label_none_size(self):
        # Set label
        d1 = caffe.io.array_to_datum(
            np.ones((10,10,3)), label=1)
        # Don't set label
        d2 = caffe.io.array_to_datum(
            np.ones((10,10,3)))
        # Not setting the label should result in a smaller object
        self.assertGreater(
            len(d1.SerializeToString()),
            len(d2.SerializeToString()))
