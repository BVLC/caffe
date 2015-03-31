import os
import tempfile
import unittest

import numpy as np

import caffe
from caffe.proto import caffe_pb2


def create_blob(shape):
    net_file = None
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            "input: 'data' input_shape { %s }" % (
                ' '.join(['dim: %d' % i for i in shape])))
        net_file = f.name
    net = caffe.Net(net_file, caffe.TRAIN)
    os.remove(net_file)
    return net.blobs['data']


class TestCreateLayer(unittest.TestCase):

    def setUp(self):
        self.shapei = [2, 2, 4, 4]
        self.blobi = create_blob(self.shapei)
        self.blobo = create_blob([1])

    def test_create_conv_layer(self):
        # Setting layer parameter for convolution
        layer_param = caffe_pb2.LayerParameter()
        layer_param.type = 'Convolution'
        layer_param.name = 'conv1'
        cparam = layer_param.convolution_param
        cparam.num_output = 3
        cparam.kernel_size = 2
        wfiller = cparam.weight_filler
        wfiller.type = "uniform"
        wfiller.max = 3
        wfiller.min = 1.5
        # Create layer
        conv_layer = caffe.create_layer(layer_param)
        self.assertEqual(conv_layer.type, 'Convolution')
        # Set up layer
        conv_layer.SetUp([self.blobi], [self.blobo])
        weights = conv_layer.blobs[0]
        self.assertTrue(np.all(weights.data >= 1.5))
        self.assertTrue(np.all(weights.data <= 3.0))
        # Reshape out blobs
        conv_layer.Reshape([self.blobi], [self.blobo])
        shapei = self.shapei
        shapeo = self.blobo.data.shape
        self.assertEqual(
            shapeo,
            (shapei[0], cparam.num_output,
                shapei[2] - cparam.kernel_size + 1,
                shapei[3] - cparam.kernel_size + 1))
        # Forward, Backward
        conv_layer.Forward([self.blobi], [self.blobo])
        conv_layer.Backward([self.blobo], [True], [self.blobi])
