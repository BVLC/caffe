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
