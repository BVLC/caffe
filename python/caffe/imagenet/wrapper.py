#!/usr/bin/env python
"""wrapper.py implements an end-to-end wrapper that classifies an image read
from disk, using the imagenet classifier.
"""

import numpy as np
import os

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
  # Move here from the global namespace to avoid overriding argparse help
  from skimage import io
  from skimage import transform
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
    self.caffenet = caffe.CaffeNet(model_def_file, pretrained_model)
    self._output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
    self._center_only = center_only

  def predict(self, filename):
    input_blob = [prepare_image(filename, self._center_only)]
    self.caffenet.Forward(input_blob, self._output_blobs)
    return self._output_blobs[0].mean(0).flatten()


def main(args):
  """
  The main function will carry out classification.
  """
  import glob
  import time
  
  net = ImageNetClassifier(args.model_def, args.pretrained_model)

  if args.gpu:
    print 'Use gpu.'
    net.caffenet.set_mode_gpu()

  files = glob.glob(os.path.join(args.root, "*." + args.ext))
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
  np.save(args.output, output)
  print 'Done. Saved to %s.' % args.output


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
    'Image classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-r', '--root',
                   help='The folder that contains images.')
  parser.add_argument('-e', '--ext', default='JPEG',
                   help='The image extension.')
  parser.add_argument('-m', '--model_def',
                   help='The model definition file.')
  parser.add_argument('-p', '--pretrained_model',
                   help='The pretrained model.')
  parser.add_argument('-o', '--output',
                   help='The output numpy file.')
  parser.add_argument('-g', '--gpu', action='store_true', default=True,
                   help='Use gpu for computation.')
  try:
    args = parser.parse_args()
    main(args)
  except:
    parser.print_help()
    exit(1)

