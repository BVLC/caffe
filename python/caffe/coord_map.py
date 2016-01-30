from __future__ import division
import numpy as np
from caffe import layers as L

PASS_THROUGH_LAYERS = ['AbsVal', 'ReLU', 'PReLU', 'Dropout', 'LRN', 'Eltwise',
        'BatchNorm', 'BNLL', 'Log', 'Exp', 'MVN', 'Power', 'Sigmoid', 'Split',
        'TanH', 'Threshold']

def conv_params(fn):
    params = fn.params.get('convolution_param', fn.params)
    axis = params.get('axis', 1)
    ks = np.array(params['kernel_size'], ndmin=1)
    dilation = np.array(params.get('dilation', 1), ndmin=1)
    assert len({'pad_h', 'pad_w', 'kernel_h', 'kernel_w', 'stride_h',
        'stride_w'} & set(fn.params)) == 0, \
                'cropping does not support legacy _h/_w params'
    return (axis, np.array(params.get('stride', 1), ndmin=1),
            (ks - 1) * dilation + 1,
            np.array(params.get('pad', 0), ndmin=1))

class UndefinedMapException(Exception):
    pass

def coord_map(fn):
    if fn.type_name in ['Convolution', 'Pooling', 'Im2col']:
        axis, stride, ks, pad = conv_params(fn)
        return axis, 1 / stride, (pad - (ks - 1) / 2) / stride
    elif fn.type_name == 'Deconvolution':
        axis, stride, ks, pad = conv_params(fn)
        return axis, stride, (ks - 1) / 2 - pad
    elif fn.type_name in PASS_THROUGH_LAYERS:
        return None, 1, 0
    elif fn.type_name == 'Crop':
        axis = fn.params.get('axis')
        return axis, 1, - fn.params['crop']
    else:
        raise UndefinedMapException

class AxisMismatchException(Exception):
    pass

def compose((ax1, a1, b1), (ax2, a2, b2)):
    if ax1 is None:
        ax = ax2
    elif ax2 is None or ax1 == ax2:
        ax = ax1
    else:
        raise AxisMismatchException
    return ax, a1 * a2, a1 * b2 + b1

def inverse((ax, a, b)):
    return ax, 1 / a, -b / a

def coord_map_from_to(top_from, top_to):
    # We need to find a common ancestor of top_from and top_to.
    # We'll assume that all ancestors are equivalent here (otherwise the graph
    # is an inconsistent state (which we could improve this to check for)).
    # For now use a brute-force algorithm.

    # walk back from top_from, keeping the coord map as we go
    from_maps = {top_from: (None, 1, 0)}
    frontier = {top_from}
    while frontier:
        top = frontier.pop()
        try:
            for bottom in top.fn.inputs:
                from_maps[bottom] = compose(from_maps[top], coord_map(top.fn))
                frontier.add(bottom)
        except UndefinedMapException:
            pass

    # now walk back from top_to until we hit a common blob
    to_maps = {top_to: (None, 1, 0)}
    frontier = {top_to}
    while frontier:
        top = frontier.pop()
        if top in from_maps:
            return compose(to_maps[top], inverse(from_maps[top]))
        try:
            for bottom in top.fn.inputs:
                to_maps[bottom] = compose(to_maps[top], coord_map(top.fn))
                frontier.add(bottom)
        except UndefinedMapException:
            continue

    # if we got here, we did not find a blob in common
    raise RuntimeError, 'Could not compute map between tops; are they connected ' \
            'by spatial layers?'

def crop(top_from, top_to):
    ax, a, b = coord_map_from_to(top_from, top_to)
    assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
    assert (b <= 0).all(), 'cannot crop negative width (b = {})'.format(b)
    assert (np.round(b) == b).all(), 'cannot crop noninteger width (b = {})'.format(b)
    return L.Crop(top_from, top_to, crop_param=dict(axis=ax, crop=list(-np.round(b).astype(int))))
