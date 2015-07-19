import unittest
import tempfile
import os
import numpy as np

import caffe

def temp_file(content):
    """Creates a file with the specified content.  Returns the filename."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        return f.name

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


def python_net_file():
    return temp_file("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'two' bottom: 'one' top: 'two'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'three' bottom: 'two' top: 'three'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }""")

DATA_LAYER_MAX = 1000
DATA_LAYER_BATCH_SIZE = 10
class SimpleDataLayer(caffe.Layer):
    """A layer that outputs the non-negative integers in order, until it reaches
       self.max-1, after which it loops back to 0 and starts over, outputting:
           0, 1, 2, 3, ..., self.max-1, 0, 1, 2, 3, ..."""

    def setup(self, bottom, top):
        self.max = DATA_LAYER_MAX
        self.batch_size = DATA_LAYER_BATCH_SIZE
        top[0].reshape(self.batch_size)
        self.reset()

    def reset(self):
        self.current_index = 0

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        next_index = self.current_index + self.batch_size
        top[0].data[...] = [x % self.max for x in range(self.current_index, next_index)]
        self.current_index = next_index % self.max
        pass

    def backward(self, top, propagate_down, bottom):
        pass

def python_data_net_file():
     return temp_file("""name: 'pythondatanet'
        layer { type: 'Python' name: 'pythondata' top: 'pythondata'
          python_param { module: 'test_python_layer' layer: 'SimpleDataLayer' } }""")


class TestPythonLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        data_net_file = python_data_net_file()
        self.data_net = caffe.Net(data_net_file, caffe.TRAIN)
        os.remove(data_net_file)

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
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)

    def test_data_forward(self):
        num_iters = 500
        for iter_num in range(num_iters):
            self.data_net.forward()
            expected_start_datum = (iter_num * DATA_LAYER_BATCH_SIZE) % DATA_LAYER_MAX
            data_range = range(expected_start_datum,
                               expected_start_datum + DATA_LAYER_BATCH_SIZE)
            expected_data = [x % DATA_LAYER_MAX for x in data_range]
            data = self.data_net.blobs['pythondata'].data
            self.assertTrue(np.all(data == expected_data))

    def test_data_reset(self):
        num_iters = 3
        batches = []
        # Run forward num_iters times, saving the batches as we go.
        for iter_num in range(num_iters):
            self.data_net.forward()
            data = self.data_net.blobs['pythondata'].data.copy()
            batches.append(data)
        # Reset the net, then run forward num_iters times again.
        # Check that we get the same results on corresponding iterations.
        self.data_net.reset()
        for previous_batch in batches:
            self.data_net.forward()
            current_batch = self.data_net.blobs['pythondata'].data
            self.assertTrue(np.all(current_batch == previous_batch))
