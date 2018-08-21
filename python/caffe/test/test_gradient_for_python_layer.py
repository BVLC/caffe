import numpy as np
import os
import caffe
from caffe import layers as L

"""
Numerically testing python layer's gradients.

How to use:
Suppose you have a python layer

  layer {
    type: "Python"
    bottom: "in_cont"
    bottom: "in_binary"
    top: "out1"
    top: "out2"
    python_param: {
      module: "folder.my_layer_module_name"
      layer: "my_layer_class_name"
      param_str: "some params"
    }
    propagate_down: true
    propagate_down: false
  }

Then you can test it's backward() gradients like this:

import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

# set the inputs
input_names_and_values = [('in_cont', np.random.randn(3,4)), ('in_binary', np.random.binomial(1, 0.4, (3,1))]
output_names = ['out1', 'out2']
py_module = 'folder.my_layer_module_name'
py_layer = 'my_layer_class_name'
param_str = 'some params'
propagate_down = [True, False]

# call the test
test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)

# you are done!


PS,
Note that you can use this utility to run gradient checks for several layers, or event entire nets.
All you have to provide is a caffe.Net object it's inputs (names and values) and outputs:

from test_gradient_for_python_layer import gradient_test_for_net

# test your net
gradient_test_for_net(net, input_names_and_values, propagate_down, output_names, thr=5e-2, h=0.25, loss_weight=2.0)

"""


def test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str=None,
                                   propagate_down=None):
    """
    Main python layer gradient test function

    Runs the test. If test fails - an AssertError is raised.

    :param input_names_and_values:  inputs to python layer, a list of tuples e.g.: [(in1_name, in1_typical_values), ...]
    :param output_names:            names of layer's top blobs, a list of names, e.g.: ['out1', 'out2',...]
    :param py_module:               value for python_param module
    :param py_layer:                value for python_param layer (python class name implementing the layer to be tested)
    :param param_str:               optional, value for python_param param_str
    :param propagate_down:          optional, list of booleans same length as len(input_names_and_shapes)

    :return: True if everything went well, if a test fails raises AssertionError detailing the failed gradient.
    """
    net, propagate_down = make_net_from_python_layer(input_names_and_values, output_names, py_module, py_layer,
                                                     param_str, propagate_down)
    h = 5e-2  # numerical gradient step size
    thr = 5.0 * h  # numerical gradient check threshold
    loss_weight = 2.0
    gradient_test_for_net(net, input_names_and_values, propagate_down, output_names, thr, h, loss_weight)
    print("done gradient test")
    return True


def make_net_from_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str=None,
                               propagate_down=None):
    """
    wrap a python layer in a "net"
    :param input_names_and_values: list of tuples [(in_name, in_data_np_array),...]
    :param output_names: names of outputs, list of strings
    :param py_module: string, "module" parameter of python layer
    :param py_layer: string, "layer" parameter for python layer
    :param param_str: optional string, "param_str" for python layer (default is None)
    :param propagate_down: list of booleans same length as len(input_names_and_shapes)

    :return:
        caffe.Net object encapsulating the tested layer
        updated propagate_down boolean vector
    """
    # build net
    ns = caffe.NetSpec()
    inputs = []
    for in_ in input_names_and_values:
        inl_ = L.DummyData(name=in_[0], dummy_data_param={'shape': {'dim': list(in_[1].shape)}})
        ns.__setattr__(in_[0], inl_)
        inputs.append(inl_)
    str(ns.to_proto())
    python_param = {'module': py_module, 'layer': py_layer}
    if param_str:
        python_param['param_str'] = param_str
    if propagate_down is None:
        propagate_down = [True for _ in xrange(len(output_names))]
    outputs = L.Python(*inputs, name='tested_py_layer', ntop=len(output_names),
                       python_param=python_param,
                       propagate_down=propagate_down,
                       loss_weight=[1.0 for _ in output_names])  # mark this layer as "loss" for gradients
    if len(output_names) == 1:
        ns.__setattr__(output_names[0], outputs)
    else:
        for o, on in zip(outputs, output_names):
            ns.__setattr__(on, o)
    with open('./test_py_layer.prototxt', 'w') as tf:
        tf.write('name: "test_py_layer"\n')
        tf.write('force_backward: true\n')  # must have. otherwise python's backward is not called at all.
        tf.write(str(ns.to_proto()))
    net = caffe.Net('./test_py_layer.prototxt', caffe.TEST)
    os.unlink('./test_py_layer.prototxt')
    return net, propagate_down


def gradient_test_for_net(net, input_names_and_values, propagate_down, output_names, thr, h, loss_weight):
    # set inputs
    for in_ in input_names_and_values:
        net.blobs[in_[0]].data[...] = in_[1].copy()
    # set parameters to random numbers
    pl = net.layers[list(net._layer_names).index('tested_py_layer')]
    p = []
    for p_ in pl.blobs:
        p_.data[...] = np.random.randn(*p_.data.shape)
        p.append(p_.data.copy())
    # test all outputs
    for out_name in output_names:
        print("\ttesting gradient of ", out_name)
        for out_i in xrange(net.blobs[out_name].data.size):
            # reset
            for in_ in input_names_and_values:
                net.blobs[in_[0]].data[...] = in_[1].copy()
            for i, p_ in enumerate(pl.blobs):
                p_.data[...] = p[i]
            test_gradient_for_specific_output_of_net(net, [in_[0] for in_ in input_names_and_values], propagate_down,
                                                     output_names, out_name, out_i, thr, h, loss_weight)


def test_gradient_for_specific_output_of_net(net, input_names, propagate_down,
                                             output_names, out_name, out_i,
                                             thr, h, loss_weight):
    """
    we consider a "loss" function that selects net.blobs[out_name].data[out_i]
    test gradients of all inputs w.r.t to this dummy loss
    """
    # get baseline gradients from layer
    net.forward()
    get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight)
    net.backward(**{o: net.blobs[o].diff for o in output_names})
    # store the computed gradients
    computed_dx = {in_name: net.blobs[in_name].diff.copy() for i, in_name in enumerate(input_names) if
                   propagate_down[i]}
    pl = net.layers[list(net._layer_names).index('tested_py_layer')]
    computed_dp = [p_.diff.copy() for p_ in pl.blobs]
    # ------------------------------------
    # compute numerical gradient for each input
    # ------------------------------------
    # for each input
    for in_name, prop_dn in zip(input_names, propagate_down):
        if prop_dn:
            print("\t\ttesting gradient w.r.t. ", in_name)
            # for each entry of x
            for xi in xrange(net.blobs[in_name].data.size):
                net.blobs[in_name].data.flat[xi] += h
                net.forward()
                pos_y = get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight)
                net.blobs[in_name].data.flat[xi] -= 2.0 * h
                net.forward()
                neg_y = get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight)
                # reset value at xi
                net.blobs[in_name].data.flat[xi] += h
                estimated_gradient = (pos_y - neg_y) / h / 2.
                computed_gradient = computed_dx[in_name].flat[xi]
                scale = np.max([1.0, np.abs(estimated_gradient), np.abs(computed_gradient)])
                assert np.abs(estimated_gradient - computed_gradient) <= thr * scale, \
                    ("Failed check for d[{}][{}]/d[{}][{}]: computed={}, estimated={}, {} > {} ({}*{})"
                     .format(out_name, out_i, in_name, xi, computed_gradient, estimated_gradient,
                             np.abs(estimated_gradient - computed_gradient), thr * scale, thr, scale))
    # for each parameter
    for i in xrange(len(pl.blobs)):
        for xi in xrange(pl.blobs[i].data.size):
            pl.blobs[i].data.flat[xi] += h
            net.forward()
            pos_y = get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight)
            pl.blobs[i].data.flat[xi] -= 2.0 * h
            net.forward()
            neg_y = get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight)
            # reset value at xi
            pl.blobs[i].data.flat[xi] += h
            estimated_gradient = (pos_y - neg_y) / h / 2.0
            computed_gradient = computed_dp[i].flat[xi]
            scale = np.max([1.0, np.abs(estimated_gradient), np.abs(computed_gradient)])
            assert np.abs(estimated_gradient - computed_gradient) <= thr * scale, \
                ("Failed check for d[{}][{}]/d[p{}][{}]: computed={}, estimated={}, {} > {} ({}*{})"
                 .format(out_name, out_i, i, xi, computed_gradient, estimated_gradient,
                         np.abs(estimated_gradient - computed_gradient), thr * scale, thr, scale))


def get_obj_and_gradient(net, output_names, out_name, out_i, loss_weight=2.0):
    """
    fake "loss": selects net.blobs[out_name].data[out_i]
    :param net:
    :param output_names:
    :param out_name:
    :param out_i:
    :return:
    """
    # null all diffs
    # set top diff according to "loss"
    for on in output_names:
        net.blobs[on].diff[...] = np.zeros(net.blobs[on].diff.shape, dtype='f4')
    loss = net.blobs[out_name].data.flat[out_i] * loss_weight
    net.blobs[out_name].diff.flat[out_i] = loss_weight
    return loss

