import unittest

import caffe
from caffe.proto import caffe_pb2


class TestLayerParameter(unittest.TestCase):

    def setUp(self):
        pass

    def test_create_from_string(self):
        lp = caffe.LayerParameter("type: 'Convolution' name: 'conv1'")

    def test_create_from_py_layer_parameter(self):
        plp = caffe_pb2.LayerParameter()
        plp.type = 'Convolution'
        plp.name = 'conv1'
        lp = caffe.LayerParameter(plp)
        plp2 = lp.to_python()
        self.assertEqual(plp, plp2)

    def test_from_python(self):
        plp = caffe_pb2.LayerParameter()
        plp.type = 'Convolution'
        plp.name = 'conv1'
        lp = caffe.LayerParameter("")
        lp.from_python(plp)
        plp2 = lp.to_python()
        self.assertEqual(plp, plp2)
