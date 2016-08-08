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

class ParameterLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.blobs.add_blob(1)
        self.blobs[0].data[0] = 0

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        self.blobs[0].diff[0] = 1

class PhaseLayer(caffe.Layer):
    """A layer for checking attribute `phase`"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bootom, top):
        top[0].reshape()

    def forward(self, bottom, top):
        top[0].data[()] = self.phase

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


def parameter_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'layer' bottom: 'data' top: 'top'
          python_param { module: 'test_python_layer' layer: 'ParameterLayer' } }
          """)
        return f.name

def phase_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        layer { type: 'Python' name: 'layer' top: 'phase'
          python_param { module: 'test_python_layer' layer: 'PhaseLayer' } }
          """)
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
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

    def test_parameter(self):
        net_file = parameter_net_file()
        net = caffe.Net(net_file, caffe.TRAIN)
        # Test forward and backward
        net.forward()
        net.backward()
        layer = net.layers[list(net._layer_names).index('layer')]
        self.assertEqual(layer.blobs[0].data[0], 0)
        self.assertEqual(layer.blobs[0].diff[0], 1)
        layer.blobs[0].data[0] += layer.blobs[0].diff[0]
        self.assertEqual(layer.blobs[0].data[0], 1)

        # Test saving and loading
        h, caffemodel_file = tempfile.mkstemp()
        net.save(caffemodel_file)
        layer.blobs[0].data[0] = -1
        self.assertEqual(layer.blobs[0].data[0], -1)
        net.copy_from(caffemodel_file)
        self.assertEqual(layer.blobs[0].data[0], 1)
        os.remove(caffemodel_file)
        
        # Test weight sharing
        net2 = caffe.Net(net_file, caffe.TRAIN)
        net2.share_with(net)
        layer = net.layers[list(net2._layer_names).index('layer')]
        self.assertEqual(layer.blobs[0].data[0], 1)

        os.remove(net_file)

    def test_phase(self):
        net_file = phase_net_file()
        for phase in caffe.TRAIN, caffe.TEST:
            net = caffe.Net(net_file, phase)
            self.assertEqual(net.forward()['phase'], phase)
