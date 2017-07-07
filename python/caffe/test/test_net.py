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
import unittest
import tempfile
import os
import numpy as np
import six
from collections import OrderedDict

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
    layer { type: 'InnerProduct' name: 'ip' bottom: 'conv' top: 'ip_blob'
      inner_product_param { num_output: """ + str(num_output) + """
        weight_filler { type: 'gaussian' std: 2.5 }
        bias_filler { type: 'constant' value: -3 } } }
    layer { type: 'SoftmaxWithLoss' name: 'loss' bottom: 'ip_blob' bottom: 'label'
      top: 'loss' }""")
    f.close()
    return f.name


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.num_output = 13
        net_file = simple_net_file(self.num_output)
        self.net = caffe.Net(net_file, caffe.TRAIN, engine='MKL2017')
        # fill in valid labels
        self.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.net.blobs['label'].data.shape)
        os.remove(net_file)

    def test_forward_backward(self):
        self.net.forward()
        self.net.backward()

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

    def test_layer_dict(self):
        layer_dict = self.net.layer_dict
        self.assertEqual(list(layer_dict.keys()), list(self.net._layer_names))
        for i, name in enumerate(self.net._layer_names):
            self.assertEqual(layer_dict[name].type,
                             self.net.layers[i].type)

    def test_forward_backward(self):
        self.net.forward()
        self.net.backward()

    def test_forward_start_end(self):
        conv_blob=self.net.blobs['conv'];
        ip_blob=self.net.blobs['ip_blob'];
        sample_data=np.random.uniform(size=conv_blob.data.shape);
        sample_data=sample_data.astype(np.float32);
        conv_blob.data[:]=sample_data;
        forward_blob=self.net.forward(start='ip',end='ip');
        self.assertIn('ip_blob',forward_blob);

        manual_forward=[];
        for i in range(0,conv_blob.data.shape[0]):
          dot=np.dot(self.net.params['ip'][0].data,
                     conv_blob.data[i].reshape(-1));
          manual_forward.append(dot+self.net.params['ip'][1].data);
        manual_forward=np.array(manual_forward);

        np.testing.assert_allclose(ip_blob.data,manual_forward,rtol=1e-3);

    def test_clear_param_diffs(self):
        # Run a forward/backward step to have non-zero diffs
        self.net.forward()
        self.net.backward()
        diff = self.net.params["conv"][0].diff
        # Check that we have non-zero diffs
        self.assertTrue(diff.max() > 0)
        self.net.clear_param_diffs()
        # Check that the diffs are now 0
        self.assertTrue((diff == 0).all())

    def test_inputs_outputs(self):
        self.assertEqual(self.net.inputs, [])
        self.assertEqual(self.net.outputs, ['loss'])

    def test_top_bottom_names(self):
        self.assertEqual(self.net.top_names,
                         OrderedDict([('data', ['data', 'label']),
                                      ('conv', ['conv']),
                                      ('ip', ['ip_blob']),
                                      ('loss', ['loss'])]))
        self.assertEqual(self.net.bottom_names,
                         OrderedDict([('data', []),
                                      ('conv', ['data']),
                                      ('ip', ['conv']),
                                      ('loss', ['ip_blob', 'label'])]))

    def test_save_and_read(self):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.close()
        self.net.save(f.name)
        net_file = simple_net_file(self.num_output)
        # Test legacy constructor
        #   should print deprecation warning
        caffe.Net(net_file, f.name, caffe.TRAIN)
        # Test named constructor
        net2 = caffe.Net(net_file, caffe.TRAIN, weights=f.name)
        os.remove(net_file)
        os.remove(f.name)
        for name in self.net.params:
            for i in range(len(self.net.params[name])):
                self.assertEqual(abs(self.net.params[name][i].data
                    - net2.params[name][i].data).sum(), 0)

    def test_save_hdf5(self):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.close()
        self.net.save_hdf5(f.name)
        net_file = simple_net_file(self.num_output)
        net2 = caffe.Net(net_file, caffe.TRAIN)
        net2.load_hdf5(f.name)
        os.remove(net_file)
        os.remove(f.name)
        for name in self.net.params:
            for i in range(len(self.net.params[name])):
                self.assertEqual(abs(self.net.params[name][i].data
                    - net2.params[name][i].data).sum(), 0)

class TestLevels(unittest.TestCase):

    TEST_NET = """
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  dummy_data_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
}
layer {
  name: "NoLevel"
  type: "InnerProduct"
  bottom: "data"
  top: "NoLevel"
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level0Only"
  type: "InnerProduct"
  bottom: "data"
  top: "Level0Only"
  include { min_level: 0 max_level: 0 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level1Only"
  type: "InnerProduct"
  bottom: "data"
  top: "Level1Only"
  include { min_level: 1 max_level: 1 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level>=0"
  type: "InnerProduct"
  bottom: "data"
  top: "Level>=0"
  include { min_level: 0 }
  inner_product_param { num_output: 1 }
}
layer {
  name: "Level>=1"
  type: "InnerProduct"
  bottom: "data"
  top: "Level>=1"
  include { min_level: 1 }
  inner_product_param { num_output: 1 }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, blobs):
        net_blobs = [b for b in net.blobs.keys() if 'data' not in b]
        self.assertEqual(net_blobs, blobs)

    def test_0(self):
        net = caffe.Net(self.f.name, caffe.TEST)
        self.check_net(net, ['NoLevel', 'Level0Only', 'Level>=0'])

    def test_1(self):
        net = caffe.Net(self.f.name, caffe.TEST, level=1)
        self.check_net(net, ['NoLevel', 'Level1Only', 'Level>=0', 'Level>=1'])


class TestStages(unittest.TestCase):

    TEST_NET = """
layer {
  name: "data"
  type: "DummyData"
  top: "data"
  dummy_data_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
}
layer {
  name: "A"
  type: "InnerProduct"
  bottom: "data"
  top: "A"
  include { stage: "A" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "B"
  type: "InnerProduct"
  bottom: "data"
  top: "B"
  include { stage: "B" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "AorB"
  type: "InnerProduct"
  bottom: "data"
  top: "AorB"
  include { stage: "A" }
  include { stage: "B" }
  inner_product_param { num_output: 1 }
}
layer {
  name: "AandB"
  type: "InnerProduct"
  bottom: "data"
  top: "AandB"
  include { stage: "A" stage: "B" }
  inner_product_param { num_output: 1 }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, blobs):
        net_blobs = [b for b in net.blobs.keys() if 'data' not in b]
        self.assertEqual(net_blobs, blobs)

    def test_A(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['A'])
        self.check_net(net, ['A', 'AorB'])

    def test_B(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['B'])
        self.check_net(net, ['B', 'AorB'])

    def test_AandB(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['A', 'B'])
        self.check_net(net, ['A', 'B', 'AorB', 'AandB'])


class TestAllInOne(unittest.TestCase):

    TEST_NET = """
layer {
  name: "train_data"
  type: "DummyData"
  top: "data"
  top: "label"
  dummy_data_param {
    shape { dim: 1 dim: 1 dim: 10 dim: 10 }
    shape { dim: 1 dim: 1 dim: 1 dim: 1 }
  }
  include { phase: TRAIN stage: "train" }
}
layer {
  name: "val_data"
  type: "DummyData"
  top: "data"
  top: "label"
  dummy_data_param {
    shape { dim: 1 dim: 1 dim: 10 dim: 10 }
    shape { dim: 1 dim: 1 dim: 1 dim: 1 }
  }
  include { phase: TEST stage: "val" }
}
layer {
  name: "deploy_data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 10 dim: 10 } }
  include { phase: TEST stage: "deploy" }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param { num_output: 2 }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
  include: { phase: TRAIN stage: "train" }
  include: { phase: TEST stage: "val" }
}
layer {
  name: "pred"
  type: "Softmax"
  bottom: "ip"
  top: "pred"
  include: { phase: TEST stage: "deploy" }
}
"""

    def setUp(self):
        self.f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.f.write(self.TEST_NET)
        self.f.close()

    def tearDown(self):
        os.remove(self.f.name)

    def check_net(self, net, outputs):
        self.assertEqual(list(net.blobs['data'].shape), [1,1,10,10])
        self.assertEqual(net.outputs, outputs)

    def test_train(self):
        net = caffe.Net(self.f.name, caffe.TRAIN, stages=['train'])
        self.check_net(net, ['loss'])

    def test_val(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['val'])
        self.check_net(net, ['loss'])

    def test_deploy(self):
        net = caffe.Net(self.f.name, caffe.TEST, stages=['deploy'])
        self.check_net(net, ['pred'])

