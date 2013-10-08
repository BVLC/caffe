"""Translates the cmrnorm layer."""
from caffe.pyutil.translator import registerer
from caffe.proto import caffe_pb2


def translator_cmrnorm(cuda_layer):
    """Translates the cmrnorm layer.
    Note: we hard-code the constant in the local response normalization
    layer to be 1. This may be different from Krizhevsky's NIPS paper but
    matches the actual cuda convnet code.
    """
    output = caffe_pb2.LayerParameter()
    output.name = cuda_layer['name']
    output.type = 'lrn'
    output.local_size = cuda_layer['size']
    output.alpha = cuda_layer['scale'] * cuda_layer['size']
    output.beta = cuda_layer['pow']
    return [output]

registerer.register_translator('cmrnorm', translator_cmrnorm)
