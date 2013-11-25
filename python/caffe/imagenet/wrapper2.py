"""
wrapper2.py classifies a number of images at once, optionally using
selective search crops.

The selective_search_ijcv_with_python code is available at
https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- [ ] store the window coordinates with the image filenames.
- [ ] batch up image filenames as well: don't want to load all of them into memory
"""
import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
from skimage import io
from skimage import transform
import selective_search_ijcv_with_python as selective_search
import caffe

IMAGE_DIM = 256
CROPPED_DIM = 227
IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

CROP_MODES = ['center_only', 'corners', 'selective_search']

# Load the imagenet mean file
IMAGENET_MEAN = np.load(
    os.path.join(os.path.dirname(__file__), 'ilsvrc_2012_mean.npy'))


def load_image(filename):
  """
  Input:
    filename: string
  Output:
    image: an image of size (256 x 256 x 3) of type uint8.
  """
  img = io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def format_image(image, window=None, dim=IMAGE_DIM):
  """
  Input:
    image: (H x W x 3) ndarray
    window: (4) ndarray
      (y, x, h, w) coordinates
    dim: int
      Final width of the square image.
  """
  # Crop a subimage if window is provided.
  if window is not None:
    image = image[
      window[0]:window[2],
      window[1]:window[3]
    ]

  # Resize to ImageNet size, convert to BGR, subtract mean.
  image = (transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255)[:, :, ::-1]
  image -= IMAGENET_MEAN

  # Resize further if needed.
  if not dim == IMAGE_DIM:
    image = transform.resize(image, (dim, dim))
  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image


def _assemble_images_center_only(image_fnames):
  all_images = []
  for image_filename in image_fnames:
    image = format_image(load_image(image_filename))
    all_images.append(np.ascontiguousarray(
        image[np.newaxis, :,
              IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
              IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM],
        dtype=np.float32
    ))
  return all_images, image_fnames


def _assemble_images_corners(image_fnames):
  all_images = []
  fnames = []
  for image_filename in image_fnames:
    image = format_image(load_image(image_filename))
    indices = [0, IMAGE_DIM - CROPPED_DIM]

    images = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for i in indices:
      for j in indices:
        images[curr] = image[:, i:i + CROPPED_DIM, j:j + CROPPED_DIM]
        curr += 1
    images[4] = image[
      :,
      IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
      IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM
    ]
    images[5:] = images[:5, :, :, ::-1]  # flipped versions

    all_images.append(images)
    fnames.append(image_filename)
  return all_images, fnames


def _assemble_images_selective_search(image_fnames):
  windows_list = selective_search.get_windows(image_fnames)
  all_images = []
  fnames = []
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    images = np.empty(
      (len(windows), 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    for i, window in enumerate(windows):
      images[i] = format_image(image, window, CROPPED_DIM)
    all_images.append(images)
    fnames.append(image_fname)
  return all_images, fnames


def assemble_batches(
  image_fnames, crop_mode='center_only', max_batch_size=256):
  """
  Assemble list of batches, each one of at most max_batch_size subimage
  objects.

  Input:
      image_fnames: list of string
      mode: string
        'center_only': the CROPPED_DIM middle of the image is taken as is
        'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
          the image, as well as their flipped versions: a total of 10 images
        'selective_search': run Selective Search region proposal on the
          image, and take each enclosing subwindow.

  Output:
      images: (X x 3 x 227 x 227) ndarray, where X <= max_batch_size
  """
  if crop_mode == 'center_only':
    all_images, fnames = _assemble_images_center_only(image_fnames)

  elif crop_mode == 'corners':
    all_images, fnames = _assemble_images_corners(image_fnames)

  elif crop_mode == 'selective_search':
    all_images, fnames = _assemble_images_selective_search(image_fnames)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Concatenate into one (N, 3, CROPPED_DIM, CROPPED_DIM) array,
  # then split to (N/max_batch_size, 3, CROPPED_DIM, CROPPED_DIM) chunks
  all_fnames = np.repeat(fnames, [len(images) for images in all_images])
  all_images = np.concatenate(all_images, 0)
  assert(all_images.shape[0] == all_fnames.shape[0])

  num_batches = 1 + int(all_images.shape[0] / max_batch_size)
  image_batches = np.array_split(all_images, num_batches, axis=0)
  fname_batches = np.array_split(all_fnames, num_batches, axis=0)
  return image_batches, fname_batches


def compute_feats(batch, layer='imagenet'):
  if layer == 'imagenet':
    num_output = 1000
  else:
    raise ValueError("Unknown layer requested: {}".format(layer))

  num = batch.shape[0]
  output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
  caffenet.Forward([batch], output_blobs)
  feats = [output_blobs[0][i].flatten() for i in range(len(output_blobs[0]))]

  return feats


if __name__ == "__main__":
  ## Parse cmdline options
  gflags.DEFINE_string(
    "model_def", "", "The model definition file.")
  gflags.DEFINE_string(
    "pretrained_model", "", "The pretrained model.")
  gflags.DEFINE_boolean(
    "gpu", False, "use gpu for computation")
  gflags.DEFINE_string(
    "crop_mode", "center_only", "Crop mode, from {}".format(CROP_MODES))
  gflags.DEFINE_string(
    "images_file", "", "File that contains image filenames.")
  gflags.DEFINE_string(
    "output", "", "The output DataFrame HDF5 filename.")
  gflags.DEFINE_string(
    "layer", "imagenet", "Layer to output.")
  FLAGS = gflags.FLAGS
  FLAGS(sys.argv)

  ## Initialize network by loading model definition and weights.
  caffenet = caffe.CaffeNet(FLAGS.model_def, FLAGS.pretrained_model)
  caffenet.set_phase_test()
  if FLAGS.gpu:
    caffenet.set_mode_gpu()

  ## Load list of image filenames and assemble into batches.
  with open(FLAGS.images_file) as f:
    image_fnames = [_.strip() for _ in f.readlines()]
  image_batches, fname_batches = assemble_batches(
    image_fnames, FLAGS.crop_mode)
  print 'Running on {} files in {} batches'.format(
    len(image_fnames), len(image_batches))

  from IPython import embed; embed()
  # TODO: debug this
  # F1125 07:17:35.980950  6826 pycaffe.cpp:52] Check failed: len(bottom) == input_blobs.size() (1 vs. 0)

  # Process the batches.
  start = time.time()
  all_feats = []
  for i, batch in range(len(image_batches)):
    if i % 10 == 0:
      print('Batch {}/{}, elapsed {:.3f} s'.format(
        i, len(image_batches), time.time() - start))
    all_feats.append(compute_feats(image_batches[i]))
  all_feats = np.concatenate(all_feats, 0)
  all_fnames = np.concatenate(fname_batches, 0)

  df = pd.DataFrame({'feat': all_feats, 'filename': all_fnames})

  # Finally, write the results.
  df.to_hdf(FLAGS.output, 'df', mode='w')
  print 'Done. Saved to %s.' % FLAGS.output
