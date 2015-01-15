from collections import OrderedDict
import re

from .proto import caffe_pb2
from google import protobuf

def uncamel(s):
    """Convert CamelCase to underscore_case."""

    return re.sub('(?!^)([A-Z])(?=[^A-Z])', r'_\1', s).lower()

def assign_proto(proto, name, val):
    if isinstance(val, list):
        getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in val.iteritems():
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)

def to_proto(tops, names):
    if not isinstance(tops, tuple):
        tops = (tops,)
    layers = OrderedDict()
    for top in tops:
        top.fn._to_proto(layers, names)

    net = caffe_pb2.NetParameter()
    net.layer.extend(layers.values())
    return net

class Top:
    def __init__(self, fn, n):
        self.fn = fn
        self.n = n

class Function:
    def __init__(self, type_name, inputs, params):
        self.type_name = type_name
        self.inputs = inputs
        self.params = params
        self.ntop = self.params.get('ntop', 1)
        if 'ntop' in self.params:
            del self.params['ntop']
        self.in_place = self.params.get('in_place', False)
        if 'in_place' in self.params:
            del self.params['in_place']
        self.tops = tuple(Top(self, n) for n in range(self.ntop))

    def _to_proto(self, layers, names):
        bottom_names = []
        for inp in self.inputs:
            if inp.fn not in layers:
                inp.fn._to_proto(layers, names)
            bottom_names.append(layers[inp.fn].top[inp.n])
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
            layer.name = names[self.tops[0]]
        else:
            for top in self.tops:
                layer.top.append(names[top])
            layer.name = layer.top[0]

        for k, v in self.params.iteritems():
            # special case to handle generic *params
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                assign_proto(getattr(layer, uncamel(self.type_name) + '_param'), k, v)

        layers[self] = layer

class Layers:
    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            if fn.ntop == 1:
                return fn.tops[0]
            else:
                return fn.tops
        return layer_fn

class Parameters:
    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()

layers = Layers()
params = Parameters()
