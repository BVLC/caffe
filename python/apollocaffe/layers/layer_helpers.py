from apollocaffe.proto import caffe_pb2

def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""

    layer = caffe_pb2.LayerParameter()
    # get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that)
    param_names = [s for s in dir(layer) if s.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # strip the final '_param' or 'Parameter'
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))

def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly."""

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

class CaffeFiller(object):
    def __init__(self, **kwargs):
        self.create(**kwargs)
    def create(self, **kwargs):
        self.filler_param = caffe_pb2.FillerParameter()
        if 'type' in kwargs:
            self.filler_param.type = kwargs['type']
        if 'value' in kwargs:
            self.filler_param.value = kwargs['value']
        if 'min' in kwargs:
            self.filler_param.min= kwargs['min']
        if 'max' in kwargs:
            self.filler_param.max = kwargs['max']
        if 'mean' in kwargs:
            self.filler_param.mean = kwargs['mean']
        if 'std' in kwargs:
            self.filler_param.std = kwargs['std']
        if 'sparse' in kwargs:
            self.filler_param.sparse = kwargs['sparse']

class Filler(CaffeFiller):
    def __init__(self, type, *args):
        if type == 'uniform':
            if isinstance(args[0], list):
                self.create(type=type, min=args[0][0], max=args[0][1])
            else:
                self.create(type=type, min=-args[0], max=args[0])
        elif type == 'constant':
            self.create(type=type, value=args[0])
        elif type == 'gaussian':
            self.create(type=type, std=args[0])
        elif type == 'xavier':
            self.create(type=type)
        else:
            raise ValueError('Filler type must be one of ["uniform", "constant", "gaussian", "xavier"]')

class Transform(object):
    def __init__(self, **kwargs):
        self.transform_param = caffe_pb2.TransformationParameter()
        if 'scale' in kwargs:
            self.transform_param.scale = kwargs['scale']
        if 'mirror' in kwargs:
            self.transform_param.mirror = kwargs['mirror']
        if 'crop_size' in kwargs:
            self.transform_param.crop_size = kwargs['crop_size']
        if 'mean_file' in kwargs:
            self.transform_param.mean_file = kwargs['mean_file']
        if 'force_color' in kwargs:
            self.transform_param.force_color = kwargs['force_color']
        if 'force_gray' in kwargs:
            self.transform_param.force_gray = kwargs['force_gray']
        if 'mean_value' in kwargs:
            for x in kwargs['mean_value']:
                self.transform_param.mean_value.append(x)

param_names = param_name_dict()
