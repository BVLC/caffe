"""Translates the convolution and group convolution layers."""
from caffe.pyutil.translator import registerer
from caffe.proto import caffe_pb2
from caffe.pyutil import convert
import numpy as np

#pylint: disable=R0914
def translator_conv(cuda_layer):
  """Translates the convolution and group convolution layers."""
  outputs = []
  output_layer = caffe_pb2.LayerParameter()
  if not cuda_layer['sharedBiases']:
    raise ValueError('Unshared bias layers not supported yet.')
  pad = -cuda_layer['padding'][0]
  if pad != 0:
    # add a padding layer first
    pad_layer = caffe_pb2.LayerParameter()
    pad_layer.name = cuda_layer['name'] + 'pad'
    pad_layer.type = 'padding'
    pad_layer.pad = pad
    outputs.append(pad_layer)
  output_layer.name = cuda_layer['name']
  output_layer.type = 'conv'
  output_layer.num_output = cuda_layer['filters']
  output_layer.group = cuda_layer['groups'][0]
  output_layer.kernelsize = cuda_layer['filterSize'][0]
  output_layer.stride = cuda_layer['stride'][0]
  # For cuda convnet, the weight is input_channels, ksize, ksize, num_kernels
  weight = cuda_layer['weights'][0].reshape(
    cuda_layer['channels'][0] / cuda_layer['groups'][0],
    cuda_layer['filterSize'][0], cuda_layer['filterSize'][0],
    cuda_layer['filters'])
  # However, our weight is organized as num_kernels, input_channels, ksize, ksize
  out_weight = weight.swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
  # The bias is simple.
  bias = cuda_layer['biases'].flatten()
  output_layer.blobs.extend(
    [convert.array_to_blobproto(out_weight),
     convert.array_to_blobproto(bias.reshape(1, 1, 1, bias.size))])
  outputs.append(output_layer)
  return outputs

registerer.register_translator('conv', translator_conv)
