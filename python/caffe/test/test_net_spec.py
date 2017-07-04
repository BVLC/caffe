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
import caffe
from caffe import layers as L
from caffe import params as P

def lenet(batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                         dict(dim=[batch_size, 1, 1, 1])],
                                  transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20,
        weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50,
        weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500,
        weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10,
        weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

def anon_lenet(batch_size):
    data, label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                     dict(dim=[batch_size, 1, 1, 1])],
                              transform_param=dict(scale=1./255), ntop=2)
    conv1 = L.Convolution(data, kernel_size=5, num_output=20,
        weight_filler=dict(type='xavier'))
    pool1 = L.Pooling(conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    conv2 = L.Convolution(pool1, kernel_size=5, num_output=50,
        weight_filler=dict(type='xavier'))
    pool2 = L.Pooling(conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    ip1 = L.InnerProduct(pool2, num_output=500,
        weight_filler=dict(type='xavier'))
    relu1 = L.ReLU(ip1, in_place=True)
    ip2 = L.InnerProduct(relu1, num_output=10,
        weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(ip2, label)
    return loss.to_proto()

def silent_net():
    n = caffe.NetSpec()
    n.data, n.data2 = L.DummyData(shape=dict(dim=3), ntop=2)
    n.silence_data = L.Silence(n.data, ntop=0)
    n.silence_data2 = L.Silence(n.data2, ntop=0)
    return n.to_proto()

class TestNetSpec(unittest.TestCase):
    def load_net(self, net_proto):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(str(net_proto))
        f.close()
        return caffe.Net(f.name, caffe.TEST)

    def test_lenet(self):
        """Construct and build the Caffe version of LeNet."""

        net_proto = lenet(50)
        # check that relu is in-place
        self.assertEqual(net_proto.layer[6].bottom,
                net_proto.layer[6].top)
        net = self.load_net(net_proto)
        # check that all layers are present
        self.assertEqual(len(net.layers), 9)

        # now the check the version with automatically-generated layer names
        net_proto = anon_lenet(50)
        self.assertEqual(net_proto.layer[6].bottom,
                net_proto.layer[6].top)
        net = self.load_net(net_proto)
        self.assertEqual(len(net.layers), 9)

    def test_zero_tops(self):
        """Test net construction for top-less layers."""

        net_proto = silent_net()
        net = self.load_net(net_proto)
        self.assertEqual(len(net.forward()), 0)

    def test_type_error(self):
        """Test that a TypeError is raised when a Function input isn't a Top."""
        data = L.DummyData(ntop=2)  # data is a 2-tuple of Tops
        r = r"^Silence input 0 is not a Top \(type is <(type|class) 'tuple'>\)$"
        with self.assertRaisesRegexp(TypeError, r):
            L.Silence(data, ntop=0)  # should raise: data is a tuple, not a Top
        L.Silence(*data, ntop=0)  # shouldn't raise: each elt of data is a Top
