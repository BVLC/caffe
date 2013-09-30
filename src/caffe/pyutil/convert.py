"""This script converts blobproto instances to numpy arrays.
"""

from caffe.proto import caffe_pb2
import numpy as np

def blobproto_to_array(blob):
  arr = np.array(blob.data).reshape(blob.num(), blob.channels(), blobs.height(),
      blobs.width())
  return arr

def array_to_blobproto(arr):
  if arr.ndim != 4:
    raise ValueError('Incorrect array shape.')
  blob = caffe_pb2.Blob()
  blob.num, blob.channels, blob.height, blob.width = arr.shape;
  blob.data.extend(arr.flat)
  return blob