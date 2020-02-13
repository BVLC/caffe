import unittest
import tempfile
import numpy as np

import caffe


def simple_memory_net_file(num_output):
    """Make a simple net prototxt using MemoryData Layer, based on test_net.cpp, returning the name
    of the (temporary) file."""

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'testnet' force_backward: true
    layer { type: 'MemoryData' name: 'data' top: 'data' top: 'label'
      memory_data_param { batch_size: 300 channels: 1 height: 3 width: 4
          } }
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


def siamese_memory_net_file(num_output):
    """Make a simple net prototxt using MemoryData Layer, based on test_net.cpp, returning the name
    of the (temporary) file."""

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'testnet' force_backward: true
    layer { type: 'MemoryData' name: 'data' top: 'data' top: 'label'
      memory_data_param { batch_size: 300 channels: 1 height: 3 width: 4
          } }
    layer { type: 'MemoryData' name: 'data_p' top: 'data_p' top: 'label_p'
      memory_data_param { batch_size: 300 channels: 1 height: 3 width: 4
          } }
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
    layer { type: 'Convolution' name: 'conv_p' bottom: 'data_p' top: 'conv_p'
      convolution_param { num_output: 11 kernel_size: 2 pad: 3
        weight_filler { type: 'gaussian' std: 1 }
        bias_filler { type: 'constant' value: 2 } }
        param { decay_mult: 1 } param { decay_mult: 0 }
        }
    layer { type: 'InnerProduct' name: 'ip_p' bottom: 'conv_p' top: 'ip_p'
      inner_product_param { num_output: """ + str(num_output) + """
        weight_filler { type: 'gaussian' std: 2.5 }
        bias_filler { type: 'constant' value: -3 } } }
    layer { type: 'ContrastiveLoss' name: 'loss' bottom: 'ip' bottom: 'ip_p' bottom: 'label'
      top: 'loss' }""")
    f.close()
    return f.name


class TestSolverMemory(unittest.TestCase):
    def setUp(self):
        self.num_output = 2
        net_f = simple_memory_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
        f.close()
        self.solver = caffe.SGDSolver(f.name)
        self.data = np.random.rand(300, 1, 3, 4).astype("float32")
        self.labels = np.random.randint(2, size=(300, 1, 1, 1)).astype("float32")
        # also make sure get_solver runs
        caffe.get_solver(f.name)
        caffe.set_mode_cpu()

    def test_forward(self):
        """ Asserting if the data is in the network"""
        # Using the layer name
        self.solver.net.set_input_arrays(self.data, self.labels, 'data')
        self.solver.net.forward()

        # Testing after forward
        np.testing.assert_array_equal(self.data,
                                      self.solver.net.blobs['data'].data)
        np.testing.assert_array_equal(self.labels,
                                      self.solver.net.blobs['label'].data)


        # Using the layer [0]
        self.solver.net.set_input_arrays(self.data, self.labels)
        self.solver.net.forward()

        # Testing after forward
        np.testing.assert_array_equal(self.data,
                                      self.solver.net.blobs['data'].data)

        np.testing.assert_array_equal(self.labels,
                                      self.solver.net.blobs['label'].data)

    def test_raise(self):
        """Catching Exceptions"""
        with self.assertRaises(RuntimeError):
            self.solver.net.set_input_arrays(self.data, self.labels, "conv")


class TestSolverSiameseMemory(unittest.TestCase):
    def setUp(self):
        self.num_output = 2
        net_f = siamese_memory_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
        f.close()
        self.solver = caffe.SGDSolver(f.name)
        self.data = np.random.rand(300, 1, 3, 4).astype("float32")
        self.data_p = np.random.rand(300, 1, 3, 4).astype("float32")
        self.labels = np.random.randint(2, size=(300, 1, 1, 1)).astype("float32")
        # also make sure get_solver runs
        caffe.get_solver(f.name)
        caffe.set_mode_cpu()

    def test_forward(self):
        """ Asserting if the data is in the network"""
        # Using the layer name
        self.solver.net.set_input_arrays(self.data, self.labels, 'data')
        self.solver.net.set_input_arrays(self.data_p, self.labels, 'data_p')
        self.solver.net.forward()

        # Testing after forward
        np.testing.assert_array_equal(self.data,
                                      self.solver.net.blobs['data'].data)
        np.testing.assert_array_equal(self.data_p,
                                      self.solver.net.blobs['data_p'].data)
        np.testing.assert_array_equal(self.labels,
                                      self.solver.net.blobs['label'].data)
