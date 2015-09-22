"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from collections import OrderedDict
try:
    from itertools import izip_longest
except:
    from itertools import zip_longest as izip_longest
import numpy as np

from ._caffe import Net, SGDSolver
import caffe.io

# We directly update methods from Net here (rather than using composition or
# inheritance) so that nets created by caffe (e.g., by SGDSolver) will
# automatically have the improved interface.


@property
def _Net_blobs(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blobs indexed by name
    """
    return OrderedDict(zip(self._blob_names, self._blobs))


@property
def _Net_blob_loss_weights(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blob loss weights indexed by name
    """
    return OrderedDict(zip(self._blob_names, self._blob_loss_weights))


@property
def _Net_params(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    parameters indexed by name; each is a list of multiple blobs (e.g.,
    weights and biases)
    """
    return OrderedDict([(name, lr.blobs)
                        for name, lr in zip(self._layer_names, self.layers)
                        if len(lr.blobs) > 0])


@property
def _Net_inputs(self):
    return [list(self.blobs.keys())[i] for i in self._inputs]


@property
def _Net_outputs(self):
    return [list(self.blobs.keys())[i] for i in self._outputs]


def _Net_forward(self, blobs=None, start=None, end=None, **kwargs):
    """
    Forward pass: prepare inputs and run the net forward.

    Parameters
    ----------
    blobs : list of blobs to return in addition to output blobs.
    kwargs : Keys are input blob names and values are blob ndarrays.
             For formatting inputs for Caffe, see Net.preprocess().
             If None, input is taken from data layers.
    start : optional name of layer at which to begin the forward pass
    end : optional name of layer at which to finish the forward pass
          (inclusive)

    Returns
    -------
    outs : {blob name: blob ndarray} dict.
    """
    if blobs is None:
        blobs = []

    if start is not None:
        start_ind = list(self._layer_names).index(start)
    else:
        start_ind = 0

    if end is not None:
        end_ind = list(self._layer_names).index(end)
        outputs = set([end] + blobs)
    else:
        end_ind = len(self.layers) - 1
        outputs = set(self.outputs + blobs)

    if kwargs:
        if set(kwargs.keys()) != set(self.inputs):
            raise Exception('Input blob arguments do not match net inputs.')
        # Set input according to defined shapes and make arrays single and
        # C-contiguous as Caffe expects.
        for in_, blob in kwargs.iteritems():
            if blob.shape[0] != self.blobs[in_].num:
                raise Exception('Input is not batch sized')
            self.blobs[in_].data[...] = blob

    self._forward(start_ind, end_ind)

    # Unpack blobs to extract
    return {out: self.blobs[out].data for out in outputs}


def _Net_backward(self, diffs=None, start=None, end=None, **kwargs):
    """
    Backward pass: prepare diffs and run the net backward.

    Parameters
    ----------
    diffs : list of diffs to return in addition to bottom diffs.
    kwargs : Keys are output blob names and values are diff ndarrays.
            If None, top diffs are taken from forward loss.
    start : optional name of layer at which to begin the backward pass
    end : optional name of layer at which to finish the backward pass
        (inclusive)

    Returns
    -------
    outs: {blob name: diff ndarray} dict.
    """
    if diffs is None:
        diffs = []

    if start is not None:
        start_ind = list(self._layer_names).index(start)
    else:
        start_ind = len(self.layers) - 1

    if end is not None:
        end_ind = list(self._layer_names).index(end)
        outputs = set([end] + diffs)
    else:
        end_ind = 0
        outputs = set(self.inputs + diffs)

    if kwargs:
        if set(kwargs.keys()) != set(self.outputs):
            raise Exception('Top diff arguments do not match net outputs.')
        # Set top diffs according to defined shapes and make arrays single and
        # C-contiguous as Caffe expects.
        for top, diff in kwargs.iteritems():
            if diff.ndim != 4:
                raise Exception('{} diff is not 4-d'.format(top))
            if diff.shape[0] != self.blobs[top].num:
                raise Exception('Diff is not batch sized')
            self.blobs[top].diff[...] = diff

    self._backward(start_ind, end_ind)

    # Unpack diffs to extract
    return {out: self.blobs[out].diff for out in outputs}


def _Net_forward_all(self, blobs=None, **kwargs):
    """
    Run net forward in batches.

    Parameters
    ----------
    blobs : list of blobs to extract as in forward()
    kwargs : Keys are input blob names and values are blob ndarrays.
             Refer to forward().

    Returns
    -------
    all_outs : {blob name: list of blobs} dict.
    """
    # Collect outputs from batches
    all_outs = {out: [] for out in set(self.outputs + (blobs or []))}
    for batch in self._batch(kwargs):
        outs = self.forward(blobs=blobs, **batch)
        for out, out_blob in outs.iteritems():
            all_outs[out].extend(out_blob.copy())
    # Package in ndarray.
    for out in all_outs:
        all_outs[out] = np.asarray(all_outs[out])
    # Discard padding.
    pad = len(all_outs.itervalues().next()) - len(kwargs.itervalues().next())
    if pad:
        for out in all_outs:
            all_outs[out] = all_outs[out][:-pad]
    return all_outs


def _Net_forward_backward_all(self, blobs=None, diffs=None, **kwargs):
    """
    Run net forward + backward in batches.

    Parameters
    ----------
    blobs: list of blobs to extract as in forward()
    diffs: list of diffs to extract as in backward()
    kwargs: Keys are input (for forward) and output (for backward) blob names
            and values are ndarrays. Refer to forward() and backward().
            Prefilled variants are called for lack of input or output blobs.

    Returns
    -------
    all_blobs: {blob name: blob ndarray} dict.
    all_diffs: {blob name: diff ndarray} dict.
    """
    # Batch blobs and diffs.
    all_outs = {out: [] for out in set(self.outputs + (blobs or []))}
    all_diffs = {diff: [] for diff in set(self.inputs + (diffs or []))}
    forward_batches = self._batch({in_: kwargs[in_]
                                   for in_ in self.inputs if in_ in kwargs})
    backward_batches = self._batch({out: kwargs[out]
                                    for out in self.outputs if out in kwargs})
    # Collect outputs from batches (and heed lack of forward/backward batches).
    for fb, bb in izip_longest(forward_batches, backward_batches, fillvalue={}):
        batch_blobs = self.forward(blobs=blobs, **fb)
        batch_diffs = self.backward(diffs=diffs, **bb)
        for out, out_blobs in batch_blobs.iteritems():
            all_outs[out].extend(out_blobs)
        for diff, out_diffs in batch_diffs.iteritems():
            all_diffs[diff].extend(out_diffs)
    # Package in ndarray.
    for out, diff in zip(all_outs, all_diffs):
        all_outs[out] = np.asarray(all_outs[out])
        all_diffs[diff] = np.asarray(all_diffs[diff])
    # Discard padding at the end and package in ndarray.
    pad = len(all_outs.itervalues().next()) - len(kwargs.itervalues().next())
    if pad:
        for out, diff in zip(all_outs, all_diffs):
            all_outs[out] = all_outs[out][:-pad]
            all_diffs[diff] = all_diffs[diff][:-pad]
    return all_outs, all_diffs


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

    Parameters
    ----------
    blobs: Keys blob names and values are lists of blobs (of any length).
           Naturally, all the lists should have the same length.

    Yields
    ------
    batch: {blob name: list of blobs} dict for a single batch.
    """
    num = len(blobs.itervalues().next())
    batch_size = self.blobs.itervalues().next().num
    remainder = num % batch_size
    num_batches = num / batch_size

    # Yield full batches.
    for b in range(num_batches):
        i = b * batch_size
        yield {name: blobs[name][i:i + batch_size] for name in blobs}

    # Yield last padded batch, if any.
    if remainder > 0:
        padded_batch = {}
        for name in blobs:
            padding = np.zeros((batch_size - remainder,)
                               + blobs[name].shape[1:])
            padded_batch[name] = np.concatenate([blobs[name][-remainder:],
                                                 padding])
        yield padded_batch

# Attach methods to Net.
Net.blobs = _Net_blobs
Net.blob_loss_weights = _Net_blob_loss_weights
Net.params = _Net_params
Net.forward = _Net_forward
Net.backward = _Net_backward
Net.forward_all = _Net_forward_all
Net.forward_backward_all = _Net_forward_backward_all
Net.set_input_arrays = _Net_set_input_arrays
Net._batch = _Net_batch
Net.inputs = _Net_inputs
Net.outputs = _Net_outputs
