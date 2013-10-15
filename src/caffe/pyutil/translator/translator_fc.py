"""translator_fc translates a fully connected layer to a decaf
InnerProductLayer.
"""
from caffe.proto import caffe_pb2
from caffe.pyutil.translator import registerer
from caffe.pyutil import convert
import numpy as np
from operator import mul

def translator_fc(cuda_layer):
  """The translator for the fc layer."""
  output_layer = caffe_pb2.LayerParameter()
  output_layer.name = cuda_layer['name']
  output_layer.type = 'innerproduct'
  output_layer.num_output = cuda_layer['outputs']

  weight = cuda_layer['weights'][0]
  weight.resize(weight.size / cuda_layer['outputs'], cuda_layer['outputs'])
  bias = cuda_layer['biases'][0].flatten()
  output_layer.blobs.extend(
      [convert.array_to_blobproto(weight.T.reshape((1,1) + weight.T.shape)),
       convert.array_to_blobproto(bias.reshape(1, 1, 1, bias.size))])
  return [output_layer]

registerer.register_translator('fc', translator_fc)
