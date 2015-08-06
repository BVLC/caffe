import unittest
import tempfile
import os
import six

import caffe


class SimpleLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = 10 * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = 10 * top[0].diff


class ExceptionLayer(caffe.Layer):
    """A layer for checking exceptions from Python"""

    def setup(self, bottom, top):
        raise RuntimeError


def python_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'two' bottom: 'one' top: 'two'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'three' bottom: 'two' top: 'three'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }""")
        return f.name


def exception_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'layer' bottom: 'data' top: 'top'
          python_param { module: 'test_python_layer' layer: 'ExceptionLayer' } }
          """)
        return f.name


class TestPythonLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward()
        for y in self.net.blobs['three'].data.flat:
            self.assertEqual(y, 10**3 * x)

    def test_backward(self):
        x = 7
        self.net.blobs['three'].diff[...] = x
        self.net.backward()
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 10**3 * x)

    def test_reshape(self):
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in six.itervalues(self.net.blobs):
            for d in blob.data.shape:
                self.assertEqual(s, d)

    def test_exception(self):
        net_file = exception_net_file()
        self.assertRaises(RuntimeError, caffe.Net, net_file, caffe.TEST)
        os.remove(net_file)
