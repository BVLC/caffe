"""This script generates the mnist train and test leveldbs used in the
test.
"""
from caffe.pyutil import convert
from decaf.layers import core_layers
import numpy as np
import leveldb

# the folder that has the MNIST data
MNIST_ROOT = 'mnist'

mnist = core_layers.MNISTDataLayer(
    rootfolder=MNIST_ROOT, name='mnist', is_training = True)
db = leveldb.LevelDB('mnist-train-leveldb')

for i in range(60000):
  datum = convert.array_to_datum((mnist._data[i] * 255).reshape(1,28,28).astype(np.uint8))
  datum.label = mnist._label[i]
  db.Put('%d' % (i), datum.SerializeToString())
del db