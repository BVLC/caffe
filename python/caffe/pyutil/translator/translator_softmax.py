"""Translates the softmax layers."""
from caffe.pyutil.translator import registerer
from caffe.proto import caffe_pb2


def translator_softmax(cuda_layer):
    """Translates the softmax layers."""
    output_layer = caffe_pb2.LayerParameter()
    output_layer.name = cuda_layer['name']
    output_layer.type = 'softmax'
    return [output_layer]

registerer.register_translator('softmax', translator_softmax)
