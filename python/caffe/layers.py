from collections import OrderedDict
import re

from .proto import caffe_pb2
from google import protobuf

def uncamel(s):
    """Convert CamelCase to underscore_case."""

    return re.sub('(?!^)([A-Z])(?=[^A-Z])', r'_\1', s).lower()

def assign_proto(proto, name, val):
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in item.iteritems():
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in val.iteritems():
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)

class Top(object):
    def __init__(self, fn, n):
        self.fn = fn
        self.n = n

class Function(object):
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

    def _get_name(self, top, names, autonames):
        if top not in names:
            n = autonames.setdefault(top.fn.type_name, 1)
            autonames[top.fn.type_name] += 1
            names[top] = uncamel(top.fn.type_name) + str(n)
        return names[top]

    def _to_proto(self, layers, names, autonames):
        if self in layers:
            return
        bottom_names = []
        for inp in self.inputs:
            inp.fn._to_proto(layers, names, autonames)
            bottom_names.append(layers[inp.fn].top[inp.n])
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
            for top in self.tops:
                layer.top.append(self._get_name(top, names, autonames))
        layer.name = self._get_name(self.tops[0], names, autonames)

        for k, v in self.params.iteritems():
            # special case to handle generic *params
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer, uncamel(self.type_name) + '_param'), k, v)
                except AttributeError:
                    assign_proto(layer, k, v)

        layers[self] = layer

class NetSpec(object):
    def __init__(self):
        super(NetSpec, self).__setattr__('tops', OrderedDict())

    def __setattr__(self, name, value):
        self.tops[name] = value

    def __getattr__(self, name):
        return self.tops[name]

    def to_proto(self):
        names = {v: k for k, v in self.tops.iteritems()}
        autonames = {}
        layers = OrderedDict()
        for name, top in self.tops.iteritems():
            top.fn._to_proto(layers, names, autonames)
        net = caffe_pb2.NetParameter()
        net.layer.extend(layers.values())
        return net

class Layers(object):
    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            if fn.ntop == 1:
                return fn.tops[0]
            else:
                return fn.tops
        return layer_fn

class Parameters(object):
    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()

layers = Layers()
params = Parameters()
