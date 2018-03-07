"""
Caffe network visualization: draw the NetParameter protobuffer.


.. note::

    This requires pydot>=1.0.2, which is not included in requirements.txt since
    it requires graphviz and other prerequisites outside the scope of the
    Caffe.
"""

from caffe.proto import caffe_pb2

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}


def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_edge_label(layer):
    """Define edge label based on layer type.
    """

    if layer.type == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layer.type == 'Convolution' or layer.type == 'Deconvolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layer.type == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
    else:
        edge_label = '""'

    return edge_label


def get_layer_lr_mult(layer):
    """Get the learning rate multipliers.

    Get the learning rate multipliers for the given layer. Assumes a
    Convolution/Deconvolution/InnerProduct layer.

    Parameters
    ----------
    layer : caffe_pb2.LayerParameter
        A Convolution, Deconvolution, or InnerProduct layer.

    Returns
    -------
    learning_rates : tuple of floats
        the learning rate multipliers for the weights and biases.
    """
    if layer.type not in ['Convolution', 'Deconvolution', 'InnerProduct']:
        raise ValueError("%s layers do not have a "
                         "learning rate multiplier" % layer.type)

    if not hasattr(layer, 'param'):
        return (1.0, 1.0)

    params = getattr(layer, 'param')

    if len(params) == 0:
        return (1.0, 1.0)

    if len(params) == 1:
        lrm0 = getattr(params[0],'lr_mult', 1.0)
        return (lrm0, 1.0)

    if len(params) == 2:
        lrm0, lrm1 = [getattr(p,'lr_mult', 1.0) for p in params]
        return (lrm0, lrm1)

    raise ValueError("Could not parse the learning rate multiplier")


def get_layer_label(layer, rankdir, display_lrm=False):
    """Define node label based on layer type.

    Parameters
    ----------
    layer : caffe_pb2.LayerParameter
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    display_lrm : boolean, optional
        If True include the learning rate multipliers in the label (default is
        False).

    Returns
    -------
    node_label : string
        A label for the current layer
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = r'\n'

    # Initializes a list of descriptors that will be concatenated into the
    # `node_label`
    descriptors_list = []
    # Add the layer's name
    descriptors_list.append(layer.name)
    # Add layer's type
    if layer.type == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        layer_type = '(%s %s)' % (layer.type,
                                  pooling_types_dict[layer.pooling_param.pool])
    else:
        layer_type = '(%s)' % layer.type
    descriptors_list.append(layer_type)

    # Describe parameters for spatial operation layers
    if layer.type in ['Convolution', 'Deconvolution', 'Pooling']:
        if layer.type == 'Pooling':
            kernel_size = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            padding = layer.pooling_param.pad
        else:
            kernel_size = layer.convolution_param.kernel_size[0] if \
                len(layer.convolution_param.kernel_size) else 1
            stride = layer.convolution_param.stride[0] if \
                len(layer.convolution_param.stride) else 1
            padding = layer.convolution_param.pad[0] if \
                len(layer.convolution_param.pad) else 0
        spatial_descriptor = separator.join([
            "kernel size: %d" % kernel_size,
            "stride: %d" % stride,
            "pad: %d" % padding,
        ])
        descriptors_list.append(spatial_descriptor)

    # Add LR multiplier for learning layers
    if display_lrm and layer.type in ['Convolution', 'Deconvolution', 'InnerProduct']:
        lrm0, lrm1 = get_layer_lr_mult(layer)
        if any([lrm0, lrm1]):
            lr_mult = "lr mult: %.1f, %.1f" % (lrm0, lrm1)
            descriptors_list.append(lr_mult)

    # Concatenate the descriptors into one label
    node_label = separator.join(descriptors_list)
    # Outer double quotes needed or else colon characters don't parse
    # properly
    node_label = '"%s"' % node_label
    return node_label


def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type.
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


def get_pydot_graph(caffe_net, rankdir, label_edges=True, phase=None, display_lrm=False):
    """Create a data structure which represents the `caffe_net`.

    Parameters
    ----------
    caffe_net : object
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    label_edges : boolean, optional
        Label the edges (default is True).
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    display_lrm : boolean, optional
        If True display the learning rate multipliers when relevant (default is
        False).

    Returns
    -------
    pydot graph object
    """
    pydot_graph = pydot.Dot(caffe_net.name if caffe_net.name else 'Net',
                            graph_type='digraph',
                            rankdir=rankdir)
    pydot_nodes = {}
    pydot_edges = []
    for layer in caffe_net.layer:
        if phase is not None:
          included = False
          if len(layer.include) == 0:
            included = True
          if len(layer.include) > 0 and len(layer.exclude) > 0:
            raise ValueError('layer ' + layer.name + ' has both include '
                             'and exclude specified.')
          for layer_phase in layer.include:
            included = included or layer_phase.phase == phase
          for layer_phase in layer.exclude:
            included = included and not layer_phase.phase == phase
          if not included:
            continue
        node_label = get_layer_label(layer, rankdir, display_lrm=display_lrm)
        node_name = "%s_%s" % (layer.name, layer.type)
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and
           layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            pydot_nodes[node_name] = pydot.Node(node_label,
                                                **NEURON_LAYER_STYLE)
        else:
            layer_style = LAYER_STYLE_DEFAULT
            layer_style['fillcolor'] = choose_color_by_layertype(layer.type)
            pydot_nodes[node_name] = pydot.Node(node_label, **layer_style)
        for bottom_blob in layer.bottom:
            pydot_nodes[bottom_blob + '_blob'] = pydot.Node('%s' % bottom_blob,
                                                            **BLOB_STYLE)
            edge_label = '""'
            pydot_edges.append({'src': bottom_blob + '_blob',
                                'dst': node_name,
                                'label': edge_label})
        for top_blob in layer.top:
            pydot_nodes[top_blob + '_blob'] = pydot.Node('%s' % (top_blob))
            if label_edges:
                edge_label = get_edge_label(layer)
            else:
                edge_label = '""'
            pydot_edges.append({'src': node_name,
                                'dst': top_blob + '_blob',
                                'label': edge_label})
    # Now, add the nodes and edges to the graph.
    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge['src']],
                       pydot_nodes[edge['dst']],
                       label=edge['label']))
    return pydot_graph


def draw_net(caffe_net, rankdir, ext='png', phase=None, display_lrm=False):
    """Draws a caffe net and returns the image string encoded using the given
    extension.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    ext : string, optional
        The image extension (the default is 'png').
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    display_lrm : boolean, optional
        If True display the learning rate multipliers for the learning layers
        (default is False).

    Returns
    -------
    string :
        Postscript representation of the graph.
    """
    return get_pydot_graph(caffe_net, rankdir, phase=phase,
                           display_lrm=display_lrm).create(format=ext)


def draw_net_to_file(caffe_net, filename, rankdir='LR', phase=None, display_lrm=False):
    """Draws a caffe net, and saves it to file using the format given as the
    file extension. Use '.raw' to output raw text that you can manually feed
    to graphviz to draw graphs.

    Parameters
    ----------
    caffe_net : a caffe.proto.caffe_pb2.NetParameter protocol buffer.
    filename : string
        The path to a file where the networks visualization will be stored.
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.
    phase : {caffe_pb2.Phase.TRAIN, caffe_pb2.Phase.TEST, None} optional
        Include layers from this network phase.  If None, include all layers.
        (the default is None)
    display_lrm : boolean, optional
        If True display the learning rate multipliers for the learning layers
        (default is False).
    """
    ext = filename[filename.rfind('.')+1:]
    with open(filename, 'wb') as fid:
        fid.write(draw_net(caffe_net, rankdir, ext, phase, display_lrm))
