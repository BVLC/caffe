"""Translates the pooling layers."""
from caffe.pyutil.translator import registerer
from caffe.proto import caffe_pb2
import math

def translator_pool(cuda_layer):
  """Translates the pooling layers."""
  output_layer = caffe_pb2.LayerParameter()
  output_layer.name = cuda_layer['name']
  output_layer.type = 'pool'
  method = cuda_layer['pool']
  if method == 'max':
    output_layer.pool = caffe_pb2.LayerParameter.MAX
  elif method == 'avg':
    output_layer.pool = caffe_pb2.LayerParameter.AVE
  else:
    raise NotImplementedError('Unrecognized pooling method: %s' % method)
  if cuda_layer['start'] != 0:
    raise NotImplementedError('Unsupported layer with a non-zero start.')
  # Check the outputsX size.
  output_size = math.ceil(
    float(cuda_layer['imgSize'] - cuda_layer['sizeX']) /
    cuda_layer['stride']) + 1
  if cuda_layer['outputsX'] != output_size:
    raise NotImplementedError('Unsupported layer with custon output size.')
  # If all checks passed, we will return our pooling layer
  output_layer.kernelsize = cuda_layer['sizeX']
  output_layer.stride = cuda_layer['stride']
  return [output_layer]

registerer.register_translator('pool', translator_pool)
