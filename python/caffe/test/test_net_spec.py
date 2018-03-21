import unittest
import tempfile
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import name_scope, arg_scope

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

def scope_lenet(batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                         dict(dim=[batch_size, 1, 1, 1])],
                                  transform_param=dict(scale=1./255), ntop=2)
    input_data = n.data
    # a bit more complicated for testing purpose
    with arg_scope(['Pooling'], kernel_size=2, stride=2, pool=P.Pooling.MAX):
        # both InnerProduct and Convolution have identical weight_filler
        with arg_scope(['InnerProduct', 'Convolution'], weight_filler=dict(type='xavier')):
            # add additional config for Convolution
            with arg_scope(['Convolution'], param=[dict(lr_mult=0.1, decay_mult=0.1)]):
                with name_scope('block1'):
                    n.conv = L.Convolution(input_data, kernel_size=5, num_output=20, name='conv')
                    block1 = n.pool = L.Pooling(n.conv, name='pool')
                # nested scope with different seperator
                with name_scope('parent', sep='_'):
                    with name_scope('block2'):
                        # override the learning rate
                        n.conv = L.Convolution(block1, kernel_size=5, num_output=50, name='conv', param=[dict(lr_mult=1, decay_mult=2)])
                        # override the pooling method
                        # sometimes we may want to let the top name equal to the scope name
                        # this can be done by assigning the top name to a dash
                        block2 = n._ = L.Pooling(n.conv, name='pool', pool=P.Pooling.AVE)

            n.ip1 = L.InnerProduct(block2, num_output=500)
            with arg_scope(['ReLU'], in_place=True):
                n.relu = L.ReLU(n.ip1)
            n.ip2 = L.InnerProduct(n.relu, num_output=10)
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

    def test_scope(self):
        net = scope_lenet(50)
        self.assertEqual(len(net.layer), 9)
        # check name scope
        # check layer name
        self.assertEqual(net.layer[1].name, 'block1/conv')
        self.assertEqual(net.layer[2].name, 'block1/pool')
        self.assertEqual(net.layer[3].name, 'parent_block2/conv')
        self.assertEqual(net.layer[4].name, 'parent_block2/pool')
        # check top name
        self.assertEqual(net.layer[1].top[0], 'block1/conv')
        self.assertEqual(net.layer[2].top[0], 'block1/pool')
        self.assertEqual(net.layer[3].top[0], 'parent_block2/conv')
        # this should be eqaul to scope name
        self.assertEqual(net.layer[4].top[0], 'parent_block2')
        # check arg scope
        self.assertEqual(net.layer[1].convolution_param.weight_filler.type, 'xavier')
        self.assertEqual(round(net.layer[1].param[0].lr_mult, 2), 0.1)
        self.assertEqual(round(net.layer[1].param[0].decay_mult, 2), 0.1)
        self.assertEqual(net.layer[2].pooling_param.kernel_size, 2)
        self.assertEqual(net.layer[2].pooling_param.stride, 2)
        self.assertEqual(net.layer[2].pooling_param.pool, P.Pooling.MAX)
        self.assertEqual(net.layer[3].convolution_param.weight_filler.type, 'xavier')
        self.assertEqual(net.layer[3].param[0].lr_mult, 1)
        self.assertEqual(net.layer[3].param[0].decay_mult, 2)
        self.assertEqual(net.layer[4].pooling_param.kernel_size, 2)
        self.assertEqual(net.layer[4].pooling_param.stride, 2)
        self.assertEqual(net.layer[4].pooling_param.pool, P.Pooling.AVE)
        self.assertEqual(net.layer[6].bottom, net.layer[6].top)


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
