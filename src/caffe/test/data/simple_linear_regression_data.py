"""This script generates the mnist train and test leveldbs used in the
test.
"""
from caffe.pyutil import convert
import numpy as np
import leveldb

db = leveldb.LevelDB('simple-linear-regression-leveldb')

for i in range(1000):
  label = np.random.randint(2) * 2 - 1
  arr = np.random.randn(2,1,1) + label
  datum = convert.array_to_datum(arr)
  datum.label = label
  db.Put('%d' % (i), datum.SerializeToString())
del db
