#!/usr/bin/env python
"""
This is the python version of the script tools/expand_net
which expands templates layers to generate a network proto_file

Usage:
    expand_net net_proto_file_in net_proto_file_out

A template layer example:

    layers {
      name: "layer_name"
      type: TEMPLATE
      template_param {
        source: "path/to/template/file"
        variable { name: "id" value: "id_value" }
      }
    }

In path/to/template/file
    ${id} will be expanded to id_value
    if no id_value is given, the line containing ${id} will be omitted

    ${id = default_value} will expanded to id_value
    or default_value if id_value is not given.
"""
import os
import re
from google.protobuf import text_format
from caffe.proto import caffe_pb2

ID_PATTERN = r'[_a-zA-Z][_a-zA-Z0-9]*'
PATTERN_EXP = r"""
(?:
 \${
    (?P<named>%(id)s)            # <named> group matches var id
    (?:\ *=\ *(?P<default>.+?))? # <default> group matches default value of var
 }
)
""" % {'id': ID_PATTERN}

PATTERN = re.compile(PATTERN_EXP, re.VERBOSE)

def substitute(template, mapping):
    # TODO: Skip comments in prototxt
    def convert(match_obj):
        named = match_obj.group('named')
        if named is not None:
            # print '%s => %s' % (match_obj.group(), mapping[named])
            try:
                return '%s' % (mapping[named],)
            except KeyError:
                if match_obj.group('default') is not None:
                    return match_obj.group('default').strip()
                else:
                    raise KeyError
        return match_obj.group()
    result = []
    for line in template.splitlines(True):
        try:
            subtituted = PATTERN.sub(convert, line)
            # print subtituted.rstrip()
        except KeyError:
            # remove the line if no value to the var is given
            continue
        result.append(subtituted)
    return '\n'.join(result)

def join_path(cwd, path):
    result_path = os.path.normpath(os.path.join(cwd, path))
    result_path = result_path.replace(os.sep, '/')
    if result_path.startswith('/'):
        result_path = result_path[1:]
    return result_path

def expand_template(net, cwd=''):
    expanded_net = caffe_pb2.NetParameter()
    expanded_net.CopyFrom(net)
    expanded_net.ClearField('layers')
    tmp_layer_counter = 0
    for layer in net.layers:
        if not layer.HasField('name'):
            layer.name = 'temp_layer_%d' % tmp_layer_counter
            tmp_layer_counter += 1
        layer.name = join_path(cwd, layer.name)

        if layer.type == layer.TEMPLATE:
            if layer.HasField('template_param'):
                # load template and fill it with values
                with open(layer.template_param.source) as template_file:
                    layer_template = template_file.read()
                layer_template = substitute(layer_template,
                                            {var.name:var.value for var in
                                             layer.template_param.variable})
                # generate real network from the specialized template network
                sub_net = caffe_pb2.NetParameter()
                text_format.Merge(layer_template, sub_net)
                # TODO: Handle recursive definition
                net = expand_template(sub_net, layer.name)
                expanded_net.layers.MergeFrom(net.layers)
                layer_template = None
            else:
                raise Exception
        else:
            # change relative names into absolute ones
            for i, bottom in enumerate(layer.bottom):
                layer.bottom[i] = join_path(cwd, bottom)
            for i, top in enumerate(layer.top):
                layer.top[i] = join_path(cwd, top)

            expanded_net.layers.add().CopyFrom(layer)

    return expanded_net

def main(argv):
    if len(argv) != 3:
        print 'Usage: %s expand_net net_proto_file_in net_proto_file_out' % \
                os.path.basename(sys.argv[0])
    else:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(sys.argv[1]).read(), net)
        net = expand_template(net)
        print 'Expanding net to %s' % sys.argv[2]
        with open(sys.argv[2], 'w') as net_file:
            net_file.write(text_format.MessageToString(net))

if __name__ == '__main__':
    import sys
    main(sys.argv)
