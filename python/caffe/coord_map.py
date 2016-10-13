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
"""
Determine spatial relationships between layers to relate their coordinates.
Coordinates are mapped from input-to-output (forward), but can
be mapped output-to-input (backward) by the inverse mapping too.
This helps crop and align feature maps among other uses.
"""

from __future__ import division
import numpy as np
from caffe import layers as L

PASS_THROUGH_LAYERS = ['AbsVal', 'BatchNorm', 'Bias', 'BNLL', 'Dropout',
                       'Eltwise', 'ELU', 'Log', 'LRN', 'Exp', 'MVN', 'Power',
                       'ReLU', 'PReLU', 'Scale', 'Sigmoid', 'Split', 'TanH',
                       'Threshold']


def conv_params(fn):
    """
    Extract the spatial parameters that determine the coordinate mapping:
    kernel size, stride, padding, and dilation.

    Implementation detail: Convolution, Deconvolution, and Im2col layers
    define these in the convolution_param message, while Pooling has its
    own fields in pooling_param. This method deals with these details to
    extract canonical parameters.
    """
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


def crop_params(fn):
    """
    Extract the crop layer parameters with defaults.
    """
    params = fn.params.get('crop_param', fn.params)
    axis = params.get('axis', 2)  # default to spatial crop for N, C, H, W
    offset = np.array(params.get('offset', 0), ndmin=1)
    return (axis, offset)


class UndefinedMapException(Exception):
    """
    Exception raised for layers that do not have a defined coordinate mapping.
    """
    pass


def coord_map(fn):
    """
    Define the coordinate mapping by its
    - axis
    - scale: output coord[i * scale] <- input_coord[i]
    - shift: output coord[i] <- output_coord[i + shift]
    s.t. the identity mapping, as for pointwise layers like ReLu, is defined by
    (None, 1, 0) since it is independent of axis and does not transform coords.
    """
    if fn.type_name in ['Convolution', 'Pooling', 'Im2col']:
        axis, stride, ks, pad = conv_params(fn)
        return axis, 1 / stride, (pad - (ks - 1) / 2) / stride
    elif fn.type_name == 'Deconvolution':
        axis, stride, ks, pad = conv_params(fn)
        return axis, stride, (ks - 1) / 2 - pad
    elif fn.type_name in PASS_THROUGH_LAYERS:
        return None, 1, 0
    elif fn.type_name == 'Crop':
        axis, offset = crop_params(fn)
        axis -= 1  # -1 for last non-coordinate dim.
        return axis, 1, - offset
    else:
        raise UndefinedMapException


class AxisMismatchException(Exception):
    """
    Exception raised for mappings with incompatible axes.
    """
    pass


def compose(base_map, next_map):
    """
    Compose a base coord map with scale a1, shift b1 with a further coord map
    with scale a2, shift b2. The scales multiply and the further shift, b2,
    is scaled by base coord scale a1.
    """
    ax1, a1, b1 = base_map
    ax2, a2, b2 = next_map
    if ax1 is None:
        ax = ax2
    elif ax2 is None or ax1 == ax2:
        ax = ax1
    else:
        raise AxisMismatchException
    return ax, a1 * a2, a1 * b2 + b1


def inverse(coord_map):
    """
    Invert a coord map by de-scaling and un-shifting;
    this gives the backward mapping for the gradient.
    """
    ax, a, b = coord_map
    return ax, 1 / a, -b / a


def coord_map_from_to(top_from, top_to):
    """
    Determine the coordinate mapping betweeen a top (from) and a top (to).
    Walk the graph to find a common ancestor while composing the coord maps for
    from and to until they meet. As a last step the from map is inverted.
    """
    # We need to find a common ancestor of top_from and top_to.
    # We'll assume that all ancestors are equivalent here (otherwise the graph
    # is an inconsistent state (which we could improve this to check for)).
    # For now use a brute-force algorithm.

    def collect_bottoms(top):
        """
        Collect the bottoms to walk for the coordinate mapping.
        The general rule is that all the bottoms of a layer can be mapped, as
        most layers have the same coordinate mapping for each bottom.
        Crop layer is a notable exception. Only the first/cropped bottom is
        mappable; the second/dimensions bottom is excluded from the walk.
        """
        bottoms = top.fn.inputs
        if top.fn.type_name == 'Crop':
            bottoms = bottoms[:1]
        return bottoms

    # walk back from top_from, keeping the coord map as we go
    from_maps = {top_from: (None, 1, 0)}
    frontier = {top_from}
    while frontier:
        top = frontier.pop()
        try:
            bottoms = collect_bottoms(top)
            for bottom in bottoms:
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
            bottoms = collect_bottoms(top)
            for bottom in bottoms:
                to_maps[bottom] = compose(to_maps[top], coord_map(top.fn))
                frontier.add(bottom)
        except UndefinedMapException:
            continue

    # if we got here, we did not find a blob in common
    raise RuntimeError('Could not compute map between tops; are they '
                       'connected by spatial layers?')


def crop(top_from, top_to):
    """
    Define a Crop layer to crop a top (from) to another top (to) by
    determining the coordinate mapping between the two and net spec'ing
    the axis and shift parameters of the crop.
    """
    ax, a, b = coord_map_from_to(top_from, top_to)
    assert (a == 1).all(), 'scale mismatch on crop (a = {})'.format(a)
    assert (b <= 0).all(), 'cannot crop negative offset (b = {})'.format(b)
    assert (np.round(b) == b).all(), 'cannot crop noninteger offset ' \
        '(b = {})'.format(b)
    return L.Crop(top_from, top_to,
                  crop_param=dict(axis=ax + 1,  # +1 for first cropping dim.
                                  offset=list(-np.round(b).astype(int))))
