import sys
sys.path.append('../../python/')
import os
from caffe.proto import caffe_pb2
import caffe.convert
from google.protobuf import text_format
import cPickle as pickle
import numpy as np

class CudaConvNetReader(object):
    def __init__(self, net, readblobs=False):
        self.name = os.path.basename(net)
        self.readblobs = readblobs

        try:
            net = pickle.load(open(net))
        except ImportError:
            # It wants the 'options' module from cuda-convnet
            # so we fake it by creating an object whose every member
            # is a class that does nothing
            faker = type('fake', (), {'__getattr__':
                                        lambda s, n: type(n, (), {})})()
            sys.modules['options'] = faker
            net = pickle.load(open(net))

        # Support either the full pickled net state
        # or just the layer list
        if isinstance(net, dict) and 'model_state' in net:
            self.net = net['model_state']['layers']
        elif isinstance(net, list):
            self.net = net
        else:
            raise Exception("Unknown cuda-convnet net type")

    neurontypemap = {'relu': 'relu',
                     'logistic': 'sigmoid',
                     'dropout': 'dropout'}

    poolmethod = {
        'max': caffe_pb2.LayerParameter.MAX,
        'avg': caffe_pb2.LayerParameter.AVE 
    }

    def read(self):
        """
        Read the cuda-convnet file and convert it to a dict that has the
        same structure as a caffe protobuf
        """
        layers = []
        for layer in self.net:
            layertype = layer['type'].split('.')[0]
            readfn = getattr(self, 'read_' + layertype)

            convertedlayer = readfn(layer)

            layerconnection = {}
            layerconnection['layer'] = convertedlayer
            layerconnection['top'] = [layer['name']]
            layerconnection['bottom'] = [l['name'] for l in 
                                                   layer.get('inputLayers', [])]

            layers.append(layerconnection)

        return {'name': self.name, 
                'layers': layers}

    def read_data(self, layer):
        return {'type': 'data',
                'name': layer['name']
                }

    def read_conv(self, layer):
        assert len(layer['groups']) == 1
        assert layer['filters'] % layer['groups'][0] == 0
        assert layer['sharedBiases'] == True

        newlayer = {'type': 'conv',
                    'name': layer['name'],
                    'num_output': layer['filters'],
                    'weight_filler': {'type': 'gaussian',
                                      'std': layer['initW'][0]},
                    'bias_filler': {'type': 'constant',
                                    'value': layer['initB']},
                    'pad': -layer['padding'][0],
                    'kernelsize': layer['filterSize'][0],
                    'group': layer['groups'][0],
                    'stride': layer['stride'][0],
                    }

        if self.readblobs:
            # shape is ((channels/group)*filterSize*filterSize, nfilters)
            # want (nfilters, channels/group, height, width)

            weights = layer['weights'][0].T
            weights = weights.reshape(layer['filters'],
                                      layer['channels'][0]/layer['groups'][0],
                                      layer['filterSize'][0],
                                      layer['filterSize'][0])

            biases = layer['biases'].flatten()
            biases = biases.reshape(1, 1, 1, len(biases))

            weightsblob = caffe.convert.array_to_blobproto(weights)
            biasesblob = caffe.convert.array_to_blobproto(biases)
            newlayer['blobs'] = [weightsblob, biasesblob]

        return newlayer

    def read_pool(self, layer):
        return {'type': 'pool',
                'name': layer['name'],
                'num_output': layer['outputs'],
                'pool': self.poolmethod[layer['pool']],
                'kernelsize': layer['sizeX'],
                'stride': layer['stride'],
                }

    def read_fc(self, layer):
        newlayer = {'type': 'innerproduct',
                    'name': layer['name'],
                    'num_output': layer['outputs'],
                    'weight_filler': {'type': 'gaussian',
                                      'std': layer['initW'][0]},
                    'bias_filler': {'type': 'constant',
                                    'value': layer['initB']},
                    }

        if self.readblobs:
            # shape is (ninputs, noutputs)
            # want (1, 1, noutputs, ninputs)
            weights = layer['weights'][0].T
            weights = weights.reshape(1, 1, layer['outputs'],
                                      layer['numInputs'][0])

            biases = layer['biases'].flatten()
            biases = biases.reshape(1, 1, 1, len(biases))

            weightsblob = caffe.convert.array_to_blobproto(weights)
            biasesblob = caffe.convert.array_to_blobproto(biases)

            newlayer['blobs'] = [weightsblob, biasesblob]

        return newlayer

    def read_softmax(self, layer):
        return {'type': 'softmax',
                'name': layer['name']}

    def read_cost(self, layer):
        # TODO recognise when combined with softmax and
        # use softmax_loss instead
        if layer['type'] == "cost.logreg":
            return {'type': 'multinomial_logistic_loss',
                    'name': layer['name']}

    def read_neuron(self, layer):
        assert layer['neuron']['type'] in self.neurontypemap.keys()
        return {'name': layer['name'],
                'type': self.neurontypemap[layer['neuron']['type']]}

    def read_cmrnorm(self, layer):
        return {'name': layer['name'],
                'type': "lrn",
                'local_size': layer['size'],
                # cuda-convnet sneakily divides by size when reading the
                # net parameter file (layer.py:1041) so correct here
                'alpha': layer['scale'] * layer['size'],
                'beta': layer['pow']
                }

    def read_rnorm(self, layer):
        # return self.read_cmrnorm(layer)
        raise NotImplementedError('rnorm not implemented')

    def read_cnorm(self, layer):
        raise NotImplementedError('cnorm not implemented')


class CudaConvNetWriter(object):
    def __init__(self, net):
        pass

    def write_data(self, layer):
        pass

    def write_conv(self, layer):
        pass

    def write_pool(self, layer):
        pass

    def write_innerproduct(self, layer):
        pass

    def write_softmax_loss(self, layer):
        pass

    def write_softmax(self, layer):
        pass

    def write_multinomial_logistic_loss(self, layer):
        pass

    def write_relu(self, layer):
        pass

    def write_sigmoid(self, layer):
        pass

    def write_dropout(self, layer):
        pass

    def write_lrn(self, layer):
        pass

def cudaconv_to_prototxt(cudanet):
    """Convert the cuda-convnet layer definition to caffe prototxt.
    Takes the filename of a pickled cuda-convnet snapshot and returns
    a string.
    """
    netdict = CudaConvNetReader(cudanet, readblobs=False).read()
    protobufnet = dict_to_protobuf(netdict)
    return text_format.MessageToString(protobufnet)

def cudaconv_to_proto(cudanet):
    """Convert a cuda-convnet pickled network (including weights)
    to a caffe protobuffer. Takes a filename of a pickled cuda-convnet
    net and returns a NetParameter protobuffer python object,
    which can then be serialized with the SerializeToString() method
    and written to a file.
    """
    netdict = CudaConvNetReader(cudanet, readblobs=True).read()
    protobufnet = dict_to_protobuf(netdict)
    return protobufnet

# adapted from https://github.com/davyzhang/dict-to-protobuf/
def list_to_protobuf(values, message):
    """parse list to protobuf message"""
    if values == []:
        pass
    elif isinstance(values[0], dict):
        #value needs to be further parsed
        for val in values:
            cmd = message.add()
            dict_to_protobuf(val, cmd)
    else:
        #value can be set
        message.extend(values)

def dict_to_protobuf(values, message=None):
    """convert dict to protobuf"""
    if message is None:
        message = caffe_pb2.NetParameter()

    for k, val in values.iteritems():
        if isinstance(val, dict):
            #value needs to be further parsed
            dict_to_protobuf(val, getattr(message, k))
        elif isinstance(val, list):
            list_to_protobuf(val, getattr(message, k))
        else:
            #value can be set
            setattr(message, k, val)

    return message
