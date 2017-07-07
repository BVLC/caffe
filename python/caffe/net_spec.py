# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Python net specification.

This module provides a way to write nets directly in Python, using a natural,
functional style. See examples/pycaffe/caffenet.py for an example.

Currently this works as a thin wrapper around the Python protobuf interface,
with layers and parameters automatically generated for the "layers" and
"params" pseudo-modules, which are actually objects using __getattr__ magic
to generate protobuf messages.

Note that when using to_proto or Top.to_proto, names of intermediate blobs will
be automatically generated. To explicitly specify blob names, use the NetSpec
class -- assign to its attributes directly to name layers, and call
NetSpec.to_proto to serialize all assigned layers.

This interface is expected to continue to evolve as Caffe gains new capabilities
for specifying nets. In particular, the automatically generated layer names
are not guaranteed to be forward-compatible.
"""

from collections import OrderedDict, Counter

from .proto import caffe_pb2
from google import protobuf
import six


def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))


def to_proto(*tops):
    """Generate a NetParameter that contains all layers needed to compute
    all arguments."""

    layers = OrderedDict()
    autonames = Counter()
    for top in tops:
        top.fn._to_proto(layers, {}, autonames)
    net = caffe_pb2.NetParameter()
    net.layer.extend(layers.values())
    return net


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""

    is_repeated_field = hasattr(getattr(proto, name), 'extend')
    if is_repeated_field and not isinstance(val, list):
        val = [val]
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in six.iteritems(item):
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in six.iteritems(val):
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)


class Top(object):
    """A Top specifies a single output blob (which could be one of several
    produced by a layer.)"""

    def __init__(self, fn, n):
        self.fn = fn
        self.n = n

    def to_proto(self):
        """Generate a NetParameter that contains all layers needed to compute
        this top."""

        return to_proto(self)

    def _update(self, params):
        self.fn._update(params)

    def _to_proto(self, layers, names, autonames):
        return self.fn._to_proto(layers, names, autonames)


class Function(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, type_name, inputs, params):
        self.type_name = type_name
        for index, input in enumerate(inputs):
            if not isinstance(input, Top):
                raise TypeError('%s input %d is not a Top (type is %s)' %
                                (type_name, index, type(input)))
        self.inputs = inputs
        self.params = params
        self.ntop = self.params.get('ntop', 1)
        # use del to make sure kwargs are not double-processed as layer params
        if 'ntop' in self.params:
            del self.params['ntop']
        self.in_place = self.params.get('in_place', False)
        if 'in_place' in self.params:
            del self.params['in_place']
        self.tops = tuple(Top(self, n) for n in range(self.ntop))

    def _get_name(self, names, autonames):
        if self not in names and self.ntop > 0:
            names[self] = self._get_top_name(self.tops[0], names, autonames)
        elif self not in names:
            autonames[self.type_name] += 1
            names[self] = self.type_name + str(autonames[self.type_name])
        return names[self]

    def _get_top_name(self, top, names, autonames):
        if top not in names:
            autonames[top.fn.type_name] += 1
            names[top] = top.fn.type_name + str(autonames[top.fn.type_name])
        return names[top]

    def _update(self, params):
        self.params.update(params)

    def _to_proto(self, layers, names, autonames):
        if self in layers:
            return
        bottom_names = []
        for inp in self.inputs:
            inp._to_proto(layers, names, autonames)
            bottom_names.append(layers[inp.fn].top[inp.n])
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
            for top in self.tops:
                layer.top.append(self._get_top_name(top, names, autonames))
        layer.name = self._get_name(names, autonames)

        for k, v in six.iteritems(self.params):
            # special case to handle generic *params
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(layer,
                        _param_names[self.type_name] + '_param'), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        layers[self] = layer


class NetSpec(object):
    """A NetSpec contains a set of Tops (assigned directly as attributes).
    Calling NetSpec.to_proto generates a NetParameter containing all of the
    layers needed to produce all of the assigned Tops, using the assigned
    names."""

    def __init__(self):
        super(NetSpec, self).__setattr__('tops', OrderedDict())

    def __setattr__(self, name, value):
        self.tops[name] = value

    def __getattr__(self, name):
        return self.tops[name]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __delitem__(self, name):
        del self.tops[name]

    def keys(self):
        keys = [k for k, v in six.iteritems(self.tops)]
        return keys

    def vals(self):
        vals = [v for k, v in six.iteritems(self.tops)]
        return vals

    def update(self, name, params):
        self.tops[name]._update(params)

    def to_proto(self):
        names = {v: k for k, v in six.iteritems(self.tops)}
        autonames = Counter()
        layers = OrderedDict()
        for name, top in six.iteritems(self.tops):
            top._to_proto(layers, names, autonames)
        net = caffe_pb2.NetParameter()
        net.layer.extend(layers.values())
        return net


class Layers(object):
    """A Layers object is a pseudo-module which generates functions that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""

    def __getattr__(self, name):
        def layer_fn(*args, **kwargs):
            fn = Function(name, args, kwargs)
            if fn.ntop == 0:
                return fn
            elif fn.ntop == 1:
                return fn.tops[0]
            else:
                return fn.tops
        return layer_fn


class Parameters(object):
    """A Parameters object is a pseudo-module which generates constants used
    in layer parameters; e.g., Parameters().Pooling.MAX is the value used
    to specify max pooling."""

    def __getattr__(self, name):
       class Param:
            def __getattr__(self, param_name):
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
       return Param()


_param_names = param_name_dict()
layers = Layers()
params = Parameters()
