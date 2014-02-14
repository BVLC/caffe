#!/usr/bin/env python
"""This script converts blobproto instances to numpy arrays.
"""

from caffe.proto import caffe_pb2
import numpy as np


def blobproto_to_array(blob, return_diff=False):
  """Convert a blob proto to an array. In default, we will just return the data,
  unless return_diff is True, in which case we will return the diff.
  """
  if return_diff:
    return np.array(blob.diff).reshape(
        blob.num, blob.channels, blob.height, blob.width)
  else:
    return np.array(blob.data).reshape(
        blob.num, blob.channels, blob.height, blob.width)


def array_to_blobproto(arr, diff=None):
  """Converts a 4-dimensional array to blob proto. If diff is given, also
  convert the diff. You need to make sure that arr and diff have the same
  shape, and this function does not do sanity check.
  """
  if arr.ndim != 4:
    raise ValueError('Incorrect array shape.')
  blob = caffe_pb2.BlobProto()
  blob.num, blob.channels, blob.height, blob.width = arr.shape;
  blob.data.extend(arr.astype(float).flat)
  if diff is not None:
    blob.diff.extend(diff.astype(float).flat)
  return blob


def arraylist_to_blobprotovecor_str(arraylist):
  """Converts a list of arrays to a serialized blobprotovec, which could be
  then passed to a network for processing.
  """
  vec = caffe_pb2.BlobProtoVector()
  vec.blobs.extend([array_to_blobproto(arr) for arr in arraylist])
  return vec.SerializeToString()


def blobprotovector_str_to_arraylist(str):
  """Converts a serialized blobprotovec to a list of arrays.
  """
  vec = caffe_pb2.BlobProtoVector()
  vec.ParseFromString(str)
  return [blobproto_to_array(blob) for blob in vec.blobs]


def array_to_datum(arr, label=0):
  """Converts a 3-dimensional array to datum. If the array has dtype uint8,
  the output data will be encoded as a string. Otherwise, the output data
  will be stored in float format.
  """
  if arr.ndim != 3:
    raise ValueError('Incorrect array shape.')
  datum = caffe_pb2.Datum()
  datum.channels, datum.height, datum.width = arr.shape
  if arr.dtype == np.uint8:
    datum.data = arr.tostring()
  else:
    datum.float_data.extend(arr.flat)
  datum.label = label
  return datum


def datum_to_array(datum):
  """Converts a datum to an array. Note that the label is not returned,
  as one can easily get it by calling datum.label.
  """
  if len(datum.data):
    return np.fromstring(datum.data, dtype = np.uint8).reshape(
        datum.channels, datum.height, datum.width)
  else:
    return np.array(datum.float_data).astype(float).reshape(
        datum.channels, datum.height, datum.width)
