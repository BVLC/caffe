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

Net.input = property(lambda self: self.blobs.values()[0])
Net.input_scale = None  # for a model that expects data = input * input_scale

Net.output = property(lambda self: self.blobs.values()[-1])

Net.mean = None  # image mean (ndarray, input dimensional or broadcastable)


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


def _Net_set_mean(self, mean_f, mode='image'):
  """
  Set the mean to subtract for data centering.

  Take
    mean_f: path to mean .npy
    mode: image = use the whole-image mean (and check dimensions)
          channel = channel constant (i.e. mean pixel instead of mean image)
  """
  mean = np.load(mean_f)
  if mode == 'image':
    if mean.shape != self.input.data.shape[1:]:
      raise Exception('The mean shape does not match the input shape.')
    self.mean = mean
  elif mode == 'channel':
    self.mean = mean.mean(1).mean(1)
  else:
    raise Exception('Mode not in {}'.format(['image', 'channel']))

Net.set_mean = _Net_set_mean


def _Net_format_image(self, image):
  """
  Format image for input to Caffe:
  - convert to single
  - reorder color to BGR
  - reshape to 1 x K x H x W

  Take
    image: (H x W x K) ndarray

  Give
    image: (K x H x W) ndarray
  """
  caf_image = image.astype(np.float32)
  if self.input_scale:
    caf_image *= self.input_scale
  caf_image = caf_image[:, :, ::-1]
  if self.mean is not None:
    caf_image -= self.mean
  caf_image = caf_image.transpose((2, 0, 1))
  caf_image = caf_image[np.newaxis, :, :, :]
  return caf_image

Net.format_image = _Net_format_image


def _Net_decaffeinate_image(self, image):
  """
  Invert Caffe formatting; see _Net_format_image().
  """
  decaf_image = image.squeeze()
  decaf_image = decaf_image.transpose((1,2,0))
  if self.mean is not None:
    decaf_image += self.mean
  decaf_image = decaf_image[:, :, ::-1]
  if self.input_scale:
    decaf_image /= self.input_scale
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
