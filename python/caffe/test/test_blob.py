import unittest
import numpy as np

import caffe

class TestBlob(unittest.TestCase):
    def setUp(self):
        pass

    def test_constructor(self):
        # empty shape blob
        b = caffe.Blob([])
        self.assertEqual(b.shape, ())
        # init with list
        b = caffe.Blob([1, 2, 3])
        self.assertEqual(b.shape, (1, 2, 3))
        a = np.random.randn(1, 2, 3)
        b.data[...] = a
        self.assertTrue(np.all(a.astype('float32') == b.data))
        # init with tuple
        b = caffe.Blob((1, 2, 3))
        self.assertEqual(b.shape, (1, 2, 3))
        # init with generator
        b = caffe.Blob(xrange(2, 6))
        self.assertEqual(b.shape, tuple(xrange(2, 6)))
