"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from collections import OrderedDict
from itertools import izip_longest
import numpy as np

from ._caffe import Net, SGDSolver

# We directly update methods from Net here (rather than using composition or
# inheritance) so that nets created by caffe (e.g., by SGDSolver) will
# automatically have the improved interface.

# Input preprocessing
Net.mean = {}       # image mean (ndarray, input dimensional or broadcastable)
Net.input_scale = {}    # for a model that expects data = input * input_scale
Net.channel_swap = {}  # for RGB -> BGR and the like


@property
def _Net_blobs(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blobs indexed by name
    """
    return OrderedDict([(bl.name, bl) for bl in self._blobs])


@property
def _Net_params(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    parameters indexed by name; each is a list of multiple blobs (e.g.,
    weights and biases)
    """
    return OrderedDict([(lr.name, lr.blobs) for lr in self.layers
                        if len(lr.blobs) > 0])


def _Net_forward(self, blobs=None, **kwargs):
    """
    Forward pass: prepare inputs and run the net forward.

    Take
    blobs: list of blobs to return in addition to output blobs.
    kwargs: Keys are input blob names and values are lists of inputs.
            Images must be (H x W x K) ndarrays.
            If None, input is taken from data layers by ForwardPrefilled().

    Give
    outs: {blob name: list of blobs ndarrays} dict.
    """
    if blobs is None:
        blobs = []

    if not kwargs:
        # Carry out prefilled forward pass and unpack output.
        self.ForwardPrefilled()
        out_blobs = [self.blobs[out].data for out in self.outputs]
    else:
        # Create input and output blobs according to net defined shapes
        # and make arrays single and C-contiguous as Caffe expects.
        in_blobs = [np.ascontiguousarray(np.concatenate(kwargs[in_]),
                                         dtype=np.float32)
                    for in_ in self.inputs]
        out_blobs = [np.empty(self.blobs[out].data.shape, dtype=np.float32)
                     for out in self.outputs]

        self.Forward(in_blobs, out_blobs)

    # Unpack blobs to extract
    outs = {}
    out_blobs.extend([self.blobs[blob].data for blob in blobs])
    out_blob_names = self.outputs + blobs
    for out, out_blob in zip(out_blob_names, out_blobs):
        outs[out] = [out_blob[ix, :, :, :]
                     for ix in range(out_blob.shape[0])]
    return outs


def _Net_backward(self, diffs=None, **kwargs):
    """
    Backward pass: prepare diffs and run the net backward.

    Take
    diffs: list of diffs to return in addition to bottom diffs.
    kwargs: Keys are output blob names and values are lists of diffs.
            If None, top diffs are taken from loss by BackwardPrefilled().

    Give
    outs: {blob name: list of diffs} dict.
    """
    if diffs is None:
        diffs = []

    if not kwargs:
        # Carry out backward with forward loss diffs and unpack bottom diffs.
        self.BackwardPrefilled()
        out_diffs = [self.blobs[in_].diff for in_ in self.inputs]
    else:
        # Create top and bottom diffs according to net defined shapes
        # and make arrays single and C-contiguous as Caffe expects.
        top_diffs = [np.ascontiguousarray(np.concatenate(kwargs[out]),
                                          dtype=np.float32)
                     for out in self.outputs]
        out_diffs = [np.empty(self.blobs[bottom].diff.shape, dtype=np.float32)
                     for bottom in self.inputs]

        self.Backward(top_diffs, out_diffs)

    # Unpack diffs to extract
    outs = {}
    out_diffs.extend([self.blobs[diff].diff for diff in diffs])
    out_diff_names = self.inputs + diffs
    for out, out_diff in zip(out_diff_names, out_diffs):
        outs[out] = [out_diff[ix, :, :, :]
                     for ix in range(out_diff.shape[0])]
    return outs


def _Net_forward_all(self, blobs=None, **kwargs):
    """
    Run net forward in batches.

    Take
    blobs: list of blobs to extract as in forward()
    kwargs: Keys are input blob names and values are lists of blobs.
            Refer to forward().

    Give
    all_outs: {blob name: list of blobs} dict.
    """
    # Collect outputs from batches
    all_outs = {out: [] for out in self.outputs + blobs}
    for batch in self._batch(kwargs):
        outs = self.forward(blobs=blobs, **batch)
        for out, out_blobs in outs.items():
            all_outs[out].extend(out_blobs)
    # Discard padding at the end.
    pad = len(all_outs.itervalues().next()) - len(kwargs.itervalues().next())
    if pad:
        for out in all_outs:
            del all_outs[out][-pad:]
    return all_outs


def _Net_forward_backward_all(self, blobs=None, diffs=None, **kwargs):
    """
    Run net forward + backward in batches.

    Take
    blobs: list of blobs to extract as in forward()
    diffs: list of diffs to extract as in backward()
    kwargs: Keys are input (for forward) and output (for backward) blob
            names and values are lists of blobs. Refer to forward() and backward().
            Prefilled variants are called for lack of input or output blobs.

    Give
    all_blobs: {blob name: list of blobs} dict.
    all_diffs: {blob name: list of diffs} dict.
    """
    # Batch blobs and diffs.
    all_outs = {out: [] for out in self.outputs + (blobs or [])}
    all_diffs = {diff: [] for diff in self.inputs + (diffs or [])}
    forward_batches = self._batch({in_: kwargs[in_]
                                   for in_ in self.inputs if in_ in kwargs})
    backward_batches = self._batch({out: kwargs[out]
                                    for out in self.outputs if out in kwargs})
    # Collect outputs from batches (and heed lack of forward/backward batches).
    for fb, bb in izip_longest(forward_batches, backward_batches, fillvalue={}):
        batch_blobs = self.forward(blobs=blobs, **fb)
        batch_diffs = self.backward(diffs=diffs, **bb)
        for out, out_blobs in batch_blobs.items():
            all_outs[out].extend(out_blobs)
        for diff, out_diffs in batch_diffs.items():
            all_diffs[diff].extend(out_diffs)
    # Discard padding at the end.
    pad = len(all_outs.itervalues().next()) - len(kwargs.itervalues().next())
    if pad:
        for out in all_outs:
            del all_outs[out][-pad:]
        for diff in all_diffs:
            del all_diffs[diff][-pad:]
    return all_outs, all_diffs


def _Net_set_mean(self, input_, mean_f, mode='image'):
    """
    Set the mean to subtract for data centering.

    Take
    input_: which input to assign this mean.
    mean_f: path to mean .npy
    mode: image = use the whole-image mean (and check dimensions)
          channel = channel constant (i.e. mean pixel instead of mean image)
    """
    if input_ not in self.inputs:
        raise Exception('Input not in {}'.format(self.inputs))
    mean = np.load(mean_f)
    if mode == 'image':
        if mean.shape != self.input.data.shape[1:]:
            raise Exception('The mean shape does not match the input shape.')
        self.mean[input_] = mean
    elif mode == 'channel':
        self.mean[input_] = mean.mean(1).mean(1)
    else:
        raise Exception('Mode not in {}'.format(['image', 'channel']))



def _Net_set_input_scale(self, input_, scale):
    """
    Set the input feature scaling factor s.t. input blob = input * scale.

    Take
    input_: which input to assign this scale factor
    scale: scale coefficient
    """
    if input_ not in self.inputs:
        raise Exception('Input not in {}'.format(self.inputs))
    self.input_scale[input_] = scale


def _Net_set_channel_swap(self, input_, order):
    """
    Set the input channel order for e.g. RGB to BGR conversion
    as needed for the reference ImageNet model.

    Take
    input_: which input to assign this channel order
    order: the order to take the channels.
           (2,1,0) maps RGB to BGR for example.
    """
    if input_ not in self.inputs:
        raise Exception('Input not in {}'.format(self.inputs))
    self.channel_swap[input_] = order


def _Net_format_image(self, input_, image):
    """
    Format image for input to Caffe:
    - convert to single
    - scale feature
    - reorder channels (for instance color to BGR)
    - subtract mean
    - reshape to 1 x K x H x W

    Take
    image: (H x W x K) ndarray

    Give
    image: (K x H x W) ndarray
    """
    caf_image = image.astype(np.float32)
    input_scale = self.input_scale.get(input_)
    channel_order = self.channel_swap.get(input_)
    mean = self.mean.get(input_)
    if input_scale:
        caf_image *= input_scale
    if channel_order:
        caf_image = caf_image[:, :, channel_order]
    if mean:
        caf_image -= mean
    caf_image = caf_image.transpose((2, 0, 1))
    caf_image = caf_image[np.newaxis, :, :, :]
    return caf_image


def _Net_decaffeinate_image(self, input_, image):
    """
    Invert Caffe formatting; see _Net_format_image().
    """
    decaf_image = image.squeeze()
    decaf_image = decaf_image.transpose((1,2,0))
    input_scale = self.input_scale.get(input_)
    channel_order = self.channel_swap.get(input_)
    mean = self.mean.get(input_)
    if mean:
        decaf_image += mean
    if channel_order:
        decaf_image = decaf_image[:, :, channel_order[::-1]]
    if input_scale:
        decaf_image /= input_scale
    return decaf_image


def _Net_set_input_arrays(self, data, labels):
    """
    Set input arrays of the in-memory MemoryDataLayer.
    (Note: this is only for networks declared with the memory data layer.)
    """
    if labels.ndim == 1:
        labels = np.ascontiguousarray(labels[:, np.newaxis, np.newaxis,
                                             np.newaxis])
    return self._set_input_arrays(data, labels)


def _Net_batch(self, blobs):
    """
    Batch blob lists according to net's batch size.

    Take
    blobs: Keys blob names and values are lists of blobs (of any length).
           Naturally, all the lists should have the same length.

    Give (yield)
    batch: {blob name: list of blobs} dict for a single batch.
    """
    num = len(blobs.itervalues().next())
    batch_size = self.blobs.itervalues().next().num
    remainder = num % batch_size
    num_batches = (num + remainder) / batch_size

    # Yield full batches.
    for b in range(num_batches-1):
        for i in [b * batch_size]:
            yield {name: blobs[name][i:i + batch_size] for name in blobs}

    # Yield last padded batch, if any.
    if remainder > 0:
        yield {name: blobs[name][-remainder:] +
                     [np.zeros_like(blobs[name][0])] * remainder
               for name in blobs}


# Attach methods to Net.
Net.blobs = _Net_blobs
Net.params = _Net_params
Net.forward = _Net_forward
Net.backward = _Net_backward
Net.forward_all = _Net_forward_all
Net.forward_backward_all = _Net_forward_backward_all
Net.set_mean = _Net_set_mean
Net.set_input_scale = _Net_set_input_scale
Net.set_channel_swap = _Net_set_channel_swap
Net.format_image = _Net_format_image
Net.decaffeinate_image = _Net_decaffeinate_image
Net.set_input_arrays = _Net_set_input_arrays
Net._batch = _Net_batch
