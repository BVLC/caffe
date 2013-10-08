"""Translates the neuron layers."""
from caffe.pyutil.translator import registerer
from caffe.proto import caffe_pb2
import logging


def translator_neuron(cuda_layer):
    """Translates the neuron layers.
    Note: not all neuron layers are supported. We only implemented those that
    are needed for imagenet.
    """
    output_layer = caffe_pb2.LayerParameter()
    output_layer.name = cuda_layer['name']
    neurontype = cuda_layer['neuron']['type']
    if neurontype == 'relu':
        output_layer.type = 'relu'
    elif neurontype == 'dropout':
        output_layer.type = 'dropout'
        output_layer.dropout_ratio = cuda_layer['neuron']['params']['d']
    else:
        raise NotImplementedError('Neuron type %s not implemented yet.'
                                  % neurontype)
    return [output_layer]

registerer.register_translator('neuron', translator_neuron)
