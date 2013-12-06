"""
Classify a number of images at once, optionally using the selective
search window proposal method.

This implementation follows
  Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
  Rich feature hierarchies for accurate object detection and semantic
  segmentation.
  http://arxiv.org/abs/1311.2524

The selective_search_ijcv_with_python code is available at
  https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- [ ] batch up image filenames as well: don't want to load all of them into memory
"""
import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe

IMAGE_DIM = 256
CROPPED_DIM = 227
IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

CROP_MODES = ['center_only', 'corners', 'selective_search']

BATCH_SIZE = 256

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
  img = skimage.io.imread(filename)
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

  Output:
    image: (3 x H x W) ndarray
      Resized to (dim, dim).
  """
  # Crop a subimage if window is provided.
  if window is not None:
    image = image[
      window[0]:window[2],
      window[1]:window[3]
    ]

  # Resize to ImageNet size, convert to BGR, subtract mean.
  image = (skimage.transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255)[:, :, ::-1]
  image -= IMAGENET_MEAN

  # Resize further if needed.
  if not dim == IMAGE_DIM:
    image = skimage.transform.resize(image, (dim, dim))
  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image


def _assemble_images_center_only(image_fnames):
  """
  For each image, square the image and crop its center.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'filename' columns.
  """
  all_images = []
  for image_filename in image_fnames:
    image = format_image(load_image(image_filename))
    all_images.append(np.ascontiguousarray(
        image[np.newaxis, :,
              IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
              IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM],
        dtype=np.float32
    ))

  images_df = pd.DataFrame({
    'image': all_images,
    'filename': image_fnames
  })
  return images_df


def _assemble_images_corners(image_fnames):
  """
  For each image, square the image and crop its center, four corners,
  and mirrored version of the above.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'filename' columns.
  """
  all_images = []
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

  images_df = pd.DataFrame({
    'image': [row for row in images for images in all_images],
    'filename': np.repeat(image_fnames, 10)
  })
  return images_df


def _assemble_images_selective_search(image_fnames):
  """
  Run Selective Search window proposals on all images, then for each
  image-window pair, extract a square crop.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'filename' columns.
  """
  windows_list = selective_search.get_windows(image_fnames)

  data = []
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    for window in windows:
      data.append({
        'image': format_image(image, window, CROPPED_DIM)[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def assemble_batches(image_fnames, crop_mode='center_only', batch_size=256):
  """
  Assemble DataFrame of image crops for feature computation.

  Input:
    image_fnames: list of string
    mode: string
      'center_only': the CROPPED_DIM middle of the image is taken as is
      'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
        the image, as well as their flipped versions: a total of 10.
      'selective_search': run Selective Search region proposal on the
        image, and take each enclosing subwindow.

  Output:
    df_batches: list of DataFrames, each one of BATCH_SIZE rows.
      Each row has 'image', 'filename', and 'window' info.
      Column 'image' contains (X x 3 x 227 x 227) ndarrays.
      Column 'filename' contains source filenames.
      If 'filename' is None, then the row is just for padding.
  """
  if crop_mode == 'center_only':
    images_df = _assemble_images_center_only(image_fnames)

  elif crop_mode == 'corners':
    images_df = _assemble_images_corners(image_fnames)

  elif crop_mode == 'selective_search':
    images_df = _assemble_images_selective_search(image_fnames)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Make sure the DataFrame has a multiple of BATCH_SIZE rows:
  # just fill the extra rows with NaN filenames and all-zero images.
  N = images_df.shape[0]
  remainder = N % BATCH_SIZE
  if remainder > 0:
    zero_image = np.zeros_like(images_df['image'].iloc[0])
    remainder_df = pd.DataFrame([{
      'filename': None,
      'image': zero_image,
      'window': [0, 0, 0, 0]
    }] * (BATCH_SIZE - remainder))
    images_df = images_df.append(remainder_df)
    N = images_df.shape[0]

  # Split into batches of BATCH_SIZE.
  ind = np.arange(N) / BATCH_SIZE
  df_batches = [images_df[ind == i] for i in range(N / BATCH_SIZE)]
  return df_batches


def compute_feats(images_df, layer='imagenet'):
  if layer == 'imagenet':
    num_output = 1000
  else:
    raise ValueError("Unknown layer requested: {}".format(layer))

  num = images_df.shape[0]
  input_blobs = [np.concatenate(images_df['image'].values)]
  output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
  print(len(input_blobs), len(output_blobs))
  print(input_blobs[0].shape, output_blobs[0].shape)

  #caffenet.Forward(input_blobs, output_blobs)
  feats = [output_blobs[0][i].flatten() for i in range(len(output_blobs[0]))]

  # Add the features and delete the images.
  del images_df['image']
  images_df['feat'] = feats
  return images_df


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
    "batch_size", 256, "Number of image crops to let through in one go")
  gflags.DEFINE_string(
    "output", "", "The output DataFrame HDF5 filename.")
  gflags.DEFINE_string(
    "layer", "imagenet", "Layer to output.")
  FLAGS = gflags.FLAGS
  FLAGS(sys.argv)

  ## Load list of image filenames and assemble into batches.
  t = time.time()
  print('Assembling batches...')
  with open(FLAGS.images_file) as f:
    image_fnames = [_.strip() for _ in f.readlines()]
  image_batches = assemble_batches(
    image_fnames, FLAGS.crop_mode, FLAGS.batch_size)
  print('{} batches assembled in {:.3f} s'.format(
    len(image_batches), time.time() - t))

  # Initialize network by loading model definition and weights.
  t = time.time()
  print("Loading Caffe model.")
  caffenet = caffe.CaffeNet(FLAGS.model_def, FLAGS.pretrained_model)
  caffenet.set_phase_test()
  if FLAGS.gpu:
    caffenet.set_mode_gpu()
  print("Caffe model loaded in {:.3f} s".format(time.time() - t))

  # Process the batches.
  t = time.time()
  print 'Processing {} files in {} batches'.format(
    len(image_fnames), len(image_batches))
  dfs_with_feats = []
  for i in range(len(image_batches)):
    if i % 10 == 0:
      print('Batch {}/{}, elapsed time: {:.3f} s'.format(
        i, len(image_batches), time.time() - t))
    dfs_with_feats.append(compute_feats(image_batches[i]))

  # Concatenate, droppping the padding rows.
  df = pd.concat(dfs_with_feats).dropna(subset=['filename'])
  print("Processing complete after {:.3f} s.".format(time.time() - t))

  # Write our the results.
  t = time.time()
  df.to_hdf(FLAGS.output, 'df', mode='w')
  print("Done. Saving to {} took {:.3f} s.".format(
    FLAGS.output, time.time() - t))
