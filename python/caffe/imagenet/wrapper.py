#!/usr/bin/env python
"""wrapper.py implements an end-to-end wrapper that classifies an image read
from disk, using the imagenet classifier.
"""

import numpy as np
import os
from skimage import io
from skimage import transform

import caffe

IMAGE_DIM = 256
CROPPED_DIM = 227

# Load the imagenet mean file
IMAGENET_MEAN = np.load(
    os.path.join(os.path.dirname(__file__), 'ilsvrc_2012_mean.npy'))


def oversample(image, center_only=False):
  """
  Oversamples an image. Currently the indices are hard coded to the
  4 corners and the center of the image, as well as their flipped ones,
  a total of 10 images.

  Input:
      image: an image of size (256 x 256 x 3) and has data type uint8.
      center_only: if True, only return the center image.
  Output:
      images: the output of size (10 x 3 x 227 x 227)
  """
  image = image.swapaxes(1, 2).swapaxes(0, 1)
  indices = [0, IMAGE_DIM - CROPPED_DIM]
  center = int(indices[1] / 2)
  if center_only:
    return np.ascontiguousarray(
        image[np.newaxis, :, center:center + CROPPED_DIM,
              center:center + CROPPED_DIM],
        dtype=np.float32)
  else:
    images = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for i in indices:
      for j in indices:
        images[curr] = image[:, i:i + CROPPED_DIM, j:j + CROPPED_DIM]
        curr += 1
    images[4] = image[:, center:center + CROPPED_DIM,
                      center:center + CROPPED_DIM]
    # flipped version
    images[5:] = images[:5, :, :, ::-1]
    return images


def prepare_image(filename, center_only=False):
  img = io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  # Resize and convert to BGR
  img_reshape = (transform.resize(img, (IMAGE_DIM,IMAGE_DIM)) * 255)[:, :, ::-1]
  # subtract main
  img_reshape -= IMAGENET_MEAN
  return oversample(img_reshape, center_only)


class ImageNetClassifier(object):
  """
  The ImageNetClassifier is a wrapper class to perform easier deployment
  of models trained on imagenet.
  """
  def __init__(self, model_def_file, pretrained_model, center_only=False,
               num_output=1000):
    if center_only:
      num = 1
    else:
      num = 10
    self.caffenet = caffe.Net(model_def_file, pretrained_model)
    self._output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
    self._center_only = center_only

  def predict(self, filename):
    input_blob = [prepare_image(filename, self._center_only)]
    self.caffenet.Forward(input_blob, self._output_blobs)
    return self._output_blobs[0].mean(0).flatten()


def main(argv):
  """
  The main function will carry out classification.
  """
  import gflags
  import glob
  import time
  gflags.DEFINE_string("root", "", "The folder that contains images.")
  gflags.DEFINE_string("ext", "JPEG", "The image extension.")
  gflags.DEFINE_string("model_def", "", "The model definition file.")
  gflags.DEFINE_string("pretrained_model", "", "The pretrained model.")
  gflags.DEFINE_string("output", "", "The output numpy file.")
  gflags.DEFINE_boolean("gpu", True, "use gpu for computation")
  FLAGS = gflags.FLAGS
  FLAGS(argv)

  net = ImageNetClassifier(FLAGS.model_def, FLAGS.pretrained_model)

  if FLAGS.gpu:
    print 'Use gpu.'
    net.caffenet.set_mode_gpu()

  files = glob.glob(os.path.join(FLAGS.root, "*." + FLAGS.ext))
  files.sort()
  print 'A total of %d files' % len(files)
  output = np.empty((len(files), net._output_blobs[0].shape[1]),
                    dtype=np.float32)
  start = time.time()
  for i, f in enumerate(files):
    output[i] = net.predict(f)
    if i % 1000 == 0 and i > 0:
      print 'Processed %d files, elapsed %.2f s' % (i, time.time() - start)
  # Finally, write the results
  np.save(FLAGS.output, output)
  print 'Done. Saved to %s.' % FLAGS.output


if __name__ == "__main__":
  import sys
  main(sys.argv)
