import unittest
import tempfile

import os

import caffe

class ElemwiseScalarLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.layer_params_ = eval(self.param_str_)
        self.value_ = self.layer_params_['value']
        if self.layer_params_['op'].lower() == 'add':
            self._forward = self._forward_add
            self._backward = self._backward_add
        elif self.layer_params_['op'].lower() == 'mul':
            self._forward = self._forward_mul
            self._backward = self._backward_mul
        else:
            raise ValueError("Unknown operation type: '%s'"
                % self.layer_params_['op'].lower())
    def _forward_add(self, bottom, top):
        top[0].data[...] = bottom[0].data + self.value_
    def _backward_add(self, bottom, propagate_down, top):
        bottom[0].diff[...] = top[0].diff
    def _forward_mul(self, bottom, top):
        top[0].data[...] = bottom[0].data * self.value_
    def _backward_mul(self, bottom, propagate_down, top):
        bottom[0].diff[...] = top[0].diff * self.value_
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height,
                bottom[0].width)
    def forward(self, bottom, top):
        self._forward(bottom, top)
    def backward(self, top, propagate_down, bottom):
        self._backward(bottom, propagate_down, top)


def python_net_file():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(r"""name: 'pythonnet' force_backward: true
    input: 'data' input_dim: 10 input_dim: 9 input_dim: 8 input_dim: 7
    layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
      python_param {
        module: 'test_python_layer_with_param_str' layer: 'ElemwiseScalarLayer'
        param_str: "{'op': 'add', 'value': 2}" } }
    layer { type: 'Python' name: 'two' bottom: 'one' top: 'two'
      python_param {
        module: 'test_python_layer_with_param_str' layer: 'ElemwiseScalarLayer'
        param_str: "{'op': 'mul', 'value': 3}" } }
    layer { type: 'Python' name: 'three' bottom: 'two' top: 'three'
      python_param {
        module: 'test_python_layer_with_param_str' layer: 'ElemwiseScalarLayer'
        param_str: "{'op': 'add', 'value': 10}" } }""")
    f.close()
    return f.name

class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward()
        for y in self.net.blobs['three'].data.flat:
            self.assertEqual(y, (x + 2) * 3 + 10)

    def test_backward(self):
        x = 7
        self.net.blobs['three'].diff[...] = x
        self.net.backward()
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 3 * x)

    def test_reshape(self):
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
