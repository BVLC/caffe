import unittest
import numpy as np

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker

class TestGradientChecker(unittest.TestCase):

    def setUp(self):
        shape = [10, 5, 1, 1]
        pred = caffe.Blob(shape)
        label = caffe.Blob(shape)
        self.rng = np.random.RandomState(313)
        pred.data[...] = self.rng.randn(*shape)
        label.data[...] = self.rng.randn(*shape)
        self.bottom = [pred, label]
        self.top = [caffe.Blob([])]

    def test_euclidean(self):
        lp = caffe_pb2.LayerParameter()
        lp.type = "EuclideanLoss"
        layer = caffe.create_layer(lp)
        layer.SetUp(self.bottom, self.top)
        layer.Reshape(self.bottom, self.top)
        layer.Forward(self.bottom, self.top)
        # manual computation
        loss = np.sum((self.bottom[0].data - self.bottom[1].data) ** 2) \
            / self.bottom[0].shape[0] / 2.0
        self.assertAlmostEqual(float(self.top[0].data), loss, 5)
        checker = GradientChecker(1e-2, 1e-2)
        checker.check_gradient_exhaustive(
            layer, self.bottom, self.top, check_bottom='all')

    def test_inner_product(self):
        lp = caffe_pb2.LayerParameter()
        lp.type = "InnerProduct"
        lp.inner_product_param.num_output = 3
        layer = caffe.create_layer(lp)
        layer.SetUp([self.bottom[0]], self.top)
        w = self.rng.randn(*layer.blobs[0].shape)
        b = self.rng.randn(*layer.blobs[1].shape)
        layer.blobs[0].data[...] = w
        layer.blobs[1].data[...] = b
        layer.Reshape([self.bottom[0]], self.top)
        layer.Forward([self.bottom[0]], self.top)
        assert np.allclose(
            self.top[0].data,
            np.dot(
                self.bottom[0].data.reshape(self.bottom[0].shape[0], -1), w.T
                ) + b
            )
        checker = GradientChecker(1e-2, 1e-1)
        checker.check_gradient_exhaustive(
            layer, [self.bottom[0]], self.top, check_bottom=[0])
