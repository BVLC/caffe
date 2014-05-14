"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from collections import OrderedDict
import numpy as np

from ._caffe import Net, SGDSolver

# we directly update methods from Net here (rather than using composition or
# inheritance) so that nets created by caffe (e.g., by SGDSolver) will
# automatically have the improved interface

# Input preprocessing
Net.mean = {}   # image mean (ndarray, input dimensional or broadcastable)
Net.input_scale = {}  # for a model that expects data = input * input_scale
Net.channel_swap = {}  # for RGB -> BGR and the like


@property
def _Net_blobs(self):
  """
  An OrderedDict (bottom to top, i.e., input to output) of network
  blobs indexed by name
  """
  return OrderedDict([(bl.name, bl) for bl in self._blobs])

Net.blobs = _Net_blobs


@property
def _Net_params(self):
  """
  An OrderedDict (bottom to top, i.e., input to output) of network
  parameters indexed by name; each is a list of multiple blobs (e.g.,
  weights and biases)
  """
  return OrderedDict([(lr.name, lr.blobs) for lr in self.layers
                                          if len(lr.blobs) > 0])

Net.params = _Net_params


def _Net_forward(self, **kwargs):
  """
  Forward pass: prepare inputs and run the net forward.

  Take
    kwargs: Keys are input blob names and values are lists of inputs.
            Images must be (H x W x K) ndarrays.
            If None, input is taken from data layers by ForwardPrefilled().

  Give
    out: {output blob name: list of output blobs} dict.
  """
  outs = {}
  if not kwargs:
    # Carry out prefilled forward pass and unpack output.
    self.ForwardPrefilled()
    out_blobs = [self.blobs[out].data for out in self.outputs]
  else:
    # Create input and output blobs according to net defined shapes
    # and make arrays single and C-contiguous as Caffe expects.
    in_blobs = [np.ascontiguousarray(np.concatenate(kwargs[in_]),
                                     dtype=np.float32) for in_ in self.inputs]
    out_blobs = [np.empty(self.blobs[out].data.shape, dtype=np.float32)
                 for out in self.outputs]

    self.Forward(in_blobs, out_blobs)

  # Unpack output blobs
  for out, out_blob in zip(self.outputs, out_blobs):
    outs[out] = [out_blob[ix, :, :, :].squeeze()
                  for ix in range(out_blob.shape[0])]
  return outs

Net.forward = _Net_forward


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

Net.set_mean = _Net_set_mean


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

Net.set_input_scale = _Net_set_input_scale


def _Net_set_channel_swap(self, input_, order):
  """
  Set the input channel order for e.g. RGB to BGR conversion
  as needed for the reference ImageNet model.

  Take
    input_: which input to assign this channel order
    order: the order to take the channels. (2,1,0) maps RGB to BGR for example.
  """
  if input_ not in self.inputs:
    raise Exception('Input not in {}'.format(self.inputs))
  self.channel_swap[input_] = order

Net.set_channel_swap = _Net_set_channel_swap


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

Net.format_image = _Net_format_image


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

Net.decaffeinate_image = _Net_decaffeinate_image


def _Net_set_input_arrays(self, data, labels):
  """
  Set input arrays of the in-memory MemoryDataLayer.
  (Note: this is only for networks declared with the memory data layer.)
  """
  if labels.ndim == 1:
    labels = np.ascontiguousarray(labels[:, np.newaxis, np.newaxis,
                                         np.newaxis])
  return self._set_input_arrays(data, labels)

Net.set_input_arrays = _Net_set_input_arrays
