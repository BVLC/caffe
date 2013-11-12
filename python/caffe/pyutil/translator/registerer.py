"""registerer is a simple module that allows one to register a custom
translator for a specific cuda layer.
"""

from caffe.proto import caffe_pb2
import logging

# DATA_TYPENAME is the typename for the data layers at cuda convnet.
DATA_TYPENAME = 'data'
# likewise, cost typename
COST_TYPENAME = 'cost'
# _TRANSLATORS is a dictionary mapping layer names to functions that does the
# actual translations.
_TRANSLATORS = {}


def register_translator(name, translator):
  """Registers a translator."""
  _TRANSLATORS[name] = translator


def translate_layer(cuda_layer):
  """Translates a cuda layer to a decaf layer. The function will return
  False if the input layer is a data layer, in which no decaf layer needs to
  be inserted.

  Input:
    cuda_layer: a cuda layer as a dictionary, produced by the cuda convnet
      code.
  Output:
    caffe_layer: the corresponding caffe layer
  """
  layertype = cuda_layer['type']
  if layertype == DATA_TYPENAME or layertype.startswith(COST_TYPENAME):
    # if the layer type is data, it is simply a data layer.
    logging.info('Ignoring layer %s (type %s)',
        cuda_layer['name'], cuda_layer['type'])
    return False
  elif layertype in _TRANSLATORS:
    logging.info('Translating layer %s (type %s)',
        cuda_layer['name'], cuda_layer['type'])
    return _TRANSLATORS[layertype](cuda_layer)
  else:
    raise TypeError('No registered translator for %s (type %s).' %
      (cuda_layer['name'], cuda_layer['type']))


def translate_cuda_network(cuda_layers):
  """Translates a list of cuda layers to a decaf net.

  Input:
    cuda_layers: a list of layers from the cuda convnet training.
  Output:
    net_param: the net parameter corresponding to the cuda net.
  """
  caffe_net = caffe_pb2.NetParameter()
  caffe_net.name = 'CaffeNet'
  provided_blobs = set()
  for cuda_layer in cuda_layers:
    if cuda_layer['type'] == DATA_TYPENAME:
      logging.error('Ignoring data layer %s' % cuda_layer['name'])
      continue
    elif cuda_layer['type'].startswith(COST_TYPENAME):
      logging.error('Ignoring cost layer %s' % cuda_layer['name'])
      continue
    logging.error('Translating layer %s' % cuda_layer['name'])
    layer_params = translate_layer(cuda_layer)
    # Now, let's figure out the inputs of the layer
    if len(cuda_layer['inputs']) != 1:
      raise ValueError('Layer %s takes more than 1 input (not supported)'
          % cuda_layer['name'])
    needs = cuda_layers[cuda_layer['inputs'][0]]['name']
    if needs not in provided_blobs:
      caffe_net.input.extend([needs])
    for layer_param in layer_params:
      caffe_net.layers.add()
      caffe_net.layers[-1].layer.CopyFrom(layer_param)
      caffe_net.layers[-1].bottom.append(needs)
      caffe_net.layers[-1].top.append(layer_param.name)
      provided_blobs.add(layer_param.name)
      needs = layer_param.name
  return caffe_net
