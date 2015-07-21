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
