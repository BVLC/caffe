import unittest
import tempfile
import os
import numpy as np
import six

import caffe


def simple_net_file(num_output):
    """Make a simple net prototxt, based on test_net.cpp, returning the name
    of the (temporary) file."""

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'testnet' force_backward: true
    layer { type: 'DummyData' name: 'data' top: 'data' top: 'label'
      dummy_data_param { num: 5 channels: 2 height: 3 width: 4
        num: 5 channels: 1 height: 1 width: 1
        data_filler { type: 'gaussian' std: 1 }
        data_filler { type: 'constant' } } }
    layer { type: 'Convolution' name: 'conv' bottom: 'data' top: 'conv'
      convolution_param { num_output: 11 kernel_size: 2 pad: 3
        weight_filler { type: 'gaussian' std: 1 }
        bias_filler { type: 'constant' value: 2 } }
        param { decay_mult: 1 } param { decay_mult: 0 }
        }
    layer { type: 'InnerProduct' name: 'ip' bottom: 'conv' top: 'ip'
      inner_product_param { num_output: """ + str(num_output) + """
        weight_filler { type: 'gaussian' std: 2.5 }
        bias_filler { type: 'constant' value: -3 } } }
    layer { type: 'SoftmaxWithLoss' name: 'loss' bottom: 'ip' bottom: 'label'
      top: 'loss' }""")
    f.close()
    return f.name


class TestNet(unittest.TestCase):
    def setUp(self):
        self.num_output = 13
        net_file = simple_net_file(self.num_output)
        self.net = caffe.Net(net_file, caffe.TRAIN)
        # fill in valid labels
        self.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.net.blobs['label'].data.shape)
        os.remove(net_file)

    def test_memory(self):
        """Check that holding onto blob data beyond the life of a Net is OK"""

        params = sum(map(list, six.itervalues(self.net.params)), [])
        blobs = self.net.blobs.values()
        del self.net

        # now sum everything (forcing all memory to be read)
        total = 0
        for p in params:
            total += p.data.sum() + p.diff.sum()
        for bl in blobs:
            total += bl.data.sum() + bl.diff.sum()

    def test_forward_backward(self):
        self.net.forward()
        self.net.backward()

    def test_inputs_outputs(self):
        self.assertEqual(self.net.inputs, [])
        self.assertEqual(self.net.outputs, ['loss'])

    def test_save_and_read(self):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.close()
        self.net.save(f.name)
        net_file = simple_net_file(self.num_output)
        net2 = caffe.Net(net_file, f.name, caffe.TRAIN)
        os.remove(net_file)
        os.remove(f.name)
        for name in self.net.params:
            for i in range(len(self.net.params[name])):
                self.assertEqual(abs(self.net.params[name][i].data
                    - net2.params[name][i].data).sum(), 0)
