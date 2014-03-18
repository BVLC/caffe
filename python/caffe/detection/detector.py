#!/usr/bin/env python
"""
Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows
  Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
  Rich feature hierarchies for accurate object detection and semantic
  segmentation.
  http://arxiv.org/abs/1311.2524

The selective_search_ijcv_with_python code is available at
  https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- refactor into class (without globals)
- get rid of imagenet mean file and just use mean pixel value
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe

NET = None

IMAGE_DIM = None
CROPPED_DIM = None
IMAGE_CENTER = None

IMAGE_MEAN = None
CROPPED_IMAGE_MEAN = None

BATCH_SIZE = None
NUM_OUTPUT = None

CROP_MODES = ['list', 'center_only', 'corners', 'selective_search']


def load_image(filename):
  """
  Input:
    filename: string

  Output:
    image: an image of size (H x W x 3) of type uint8.
  """
  img = skimage.io.imread(filename)
  if img.ndim == 2:
    img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
  elif img.shape[2] == 4:
    img = img[:, :, :3]
  return img


def format_image(image, window=None, cropped_size=False):
  """
  Input:
    image: (H x W x 3) ndarray
    window: (4) ndarray
      (ymin, xmin, ymax, xmax) coordinates, 0-indexed
    cropped_size: bool
      Whether to output cropped size image or full size image.

  Output:
    image: (3 x H x W) ndarray
      Resized to either IMAGE_DIM or CROPPED_DIM.
    dims: (H, W) of the original image
  """
  dims = image.shape[:2]

  # Crop a subimage if window is provided.
  if window is not None:
    image = image[window[0]:window[2], window[1]:window[3]]

  # Resize to input size, subtract mean, convert to BGR
  image = image[:, :, ::-1]
  if cropped_size:
    image = skimage.transform.resize(image, (CROPPED_DIM, CROPPED_DIM)) * 255
    image -= CROPPED_IMAGE_MEAN
  else:
    image = skimage.transform.resize(image, (IMAGE_DIM, IMAGE_DIM)) * 255
    image -= IMAGE_MEAN

  image = image.swapaxes(1, 2).swapaxes(0, 1)
  return image, dims


def _image_coordinates(dims, window):
  """
  Calculate the original image coordinates of a
  window in the canonical (IMAGE_DIM x IMAGE_DIM) coordinates

  Input:
    dims: (H, W) of the original image
    window: (ymin, xmin, ymax, xmax) in the (IMAGE_DIM x IMAGE_DIM) frame

  Output:
    image_window: (ymin, xmin, ymax, xmax) in the original image frame
  """
  h, w = dims
  max_dim = float(IMAGE_DIM)
  h_scale, w_scale = h / max_dim, w / max_dim
  image_window = window * np.array((1. / h_scale, 1. / w_scale,
                                   h_scale, w_scale))
  return image_window.round().astype(int)


def _assemble_images_list(input_df):
  """
  For each image, collect the crops for the given windows.

  Input:
    input_df: pandas.DataFrame
      with 'filename', 'ymin', 'xmin', 'ymax', 'xmax' columns

  Output:
    images_df: pandas.DataFrame
      with 'image', 'window', 'filename' columns
  """
  # unpack sequence of (image filename, windows)
  coords = ['ymin', 'xmin', 'ymax', 'xmax']
  image_windows = (
    (ix, input_df.iloc[np.where(input_df.index == ix)][coords].values)
    for ix in input_df.index.unique()
  )

  # extract windows
  data = []
  for image_fname, windows in image_windows:
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_center_only(image_fnames):
  """
  For each image, square the image and crop its center.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  crop_start, crop_end = IMAGE_CENTER, IMAGE_CENTER + CROPPED_DIM
  crop_window = np.array((crop_start, crop_start, crop_end, crop_end))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    data.append({
      'image': image[np.newaxis, :,
                     crop_start:crop_end,
                     crop_start:crop_end],
      'window': _image_coordinates(dims, crop_window),
      'filename': image_fname
    })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_corners(image_fnames):
  """
  For each image, square the image and crop its center, four corners,
  and mirrored version of the above.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  # make crops
  indices = [0, IMAGE_DIM - CROPPED_DIM]
  crops = np.empty((5, 4), dtype=int)
  curr = 0
  for i in indices:
    for j in indices:
      crops[curr] = (i, j, i + CROPPED_DIM, j + CROPPED_DIM)
      curr += 1
  crops[4] = (IMAGE_CENTER, IMAGE_CENTER,
              IMAGE_CENTER + CROPPED_DIM, IMAGE_CENTER + CROPPED_DIM)
  all_crops = np.tile(crops, (2, 1))

  data = []
  for image_fname in image_fnames:
    image, dims = format_image(load_image(image_fname))
    image_crops = np.empty((10, 3, CROPPED_DIM, CROPPED_DIM), dtype=np.float32)
    curr = 0
    for crop in crops:
      image_crops[curr] = image[:, crop[0]:crop[2], crop[1]:crop[3]]
      curr += 1
    image_crops[5:] = image_crops[:5, :, :, ::-1]  # flip for mirrors
    for i in range(len(all_crops)):
      data.append({
        'image': image_crops[i][np.newaxis, :],
        'window': _image_coordinates(dims, all_crops[i]),
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def _assemble_images_selective_search(image_fnames):
  """
  Run Selective Search window proposals on all images, then for each
  image-window pair, extract a square crop.

  Input:
    image_fnames: list

  Output:
    images_df: pandas.DataFrame
      With 'image', 'window', 'filename' columns.
  """
  windows_list = selective_search.get_windows(image_fnames)

  data = []
  for image_fname, windows in zip(image_fnames, windows_list):
    image = load_image(image_fname)
    for window in windows:
      window_image, _ = format_image(image, window, cropped_size=True)
      data.append({
        'image': window_image[np.newaxis, :],
        'window': window,
        'filename': image_fname
      })

  images_df = pd.DataFrame(data)
  return images_df


def assemble_batches(inputs, crop_mode='center_only'):
  """
  Assemble DataFrame of image crops for feature computation.

  Input:
    inputs: list of filenames (center_only, corners, and selective_search mode)
      OR input DataFrame (list mode)
    mode: string
      'list': take the image windows from the input as-is
      'center_only': take the CROPPED_DIM middle of the image windows
      'corners': take CROPPED_DIM-sized boxes at 4 corners and center of
        the image windows, as well as their flipped versions: a total of 10.
      'selective_search': run Selective Search region proposal on the
        image windows, and take each enclosing subwindow.

  Output:
    df_batches: list of DataFrames, each one of BATCH_SIZE rows.
      Each row has 'image', 'filename', and 'window' info.
      Column 'image' contains (X x 3 x CROPPED_DIM x CROPPED_IM) ndarrays.
      Column 'filename' contains source filenames.
      Column 'window' contains [ymin, xmin, ymax, xmax] ndarrays.
      If 'filename' is None, then the row is just for padding.

  Note: for increased efficiency, increase the batch size (to the limit of gpu
  memory) to avoid the communication cost
  """
  if crop_mode == 'list':
    images_df = _assemble_images_list(inputs)

  elif crop_mode == 'center_only':
    images_df = _assemble_images_center_only(inputs)

  elif crop_mode == 'corners':
    images_df = _assemble_images_corners(inputs)

  elif crop_mode == 'selective_search':
    images_df = _assemble_images_selective_search(inputs)

  else:
    raise Exception("Unknown mode: not in {}".format(CROP_MODES))

  # Make sure the DataFrame has a multiple of BATCH_SIZE rows:
  # just fill the extra rows with NaN filenames and all-zero images.
  N = images_df.shape[0]
  remainder = N % BATCH_SIZE
  if remainder > 0:
    zero_image = np.zeros_like(images_df['image'].iloc[0])
    zero_window = np.zeros((1, 4), dtype=int)
    remainder_df = pd.DataFrame([{
      'filename': None,
      'image': zero_image,
      'window': zero_window
    }] * (BATCH_SIZE - remainder))
    images_df = images_df.append(remainder_df)
    N = images_df.shape[0]

  # Split into batches of BATCH_SIZE.
  ind = np.arange(N) / BATCH_SIZE
  df_batches = [images_df[ind == i] for i in range(N / BATCH_SIZE)]
  return df_batches


def compute_feats(images_df):
  input_blobs = [np.ascontiguousarray(
    np.concatenate(images_df['image'].values), dtype='float32')]
  output_blobs = [np.empty((BATCH_SIZE, NUM_OUTPUT, 1, 1), dtype=np.float32)]

  NET.Forward(input_blobs, output_blobs)
  feats = [output_blobs[0][i].flatten() for i in range(len(output_blobs[0]))]

  # Add the features and delete the images.
  del images_df['image']
  images_df['feat'] = feats
  return images_df


def config(model_def, pretrained_model, gpu, image_dim, image_mean_file):
  global IMAGE_DIM, CROPPED_DIM, IMAGE_CENTER, IMAGE_MEAN, CROPPED_IMAGE_MEAN
  global NET, BATCH_SIZE, NUM_OUTPUT

  # Initialize network by loading model definition and weights.
  t = time.time()
  print("Loading Caffe model.")
  NET = caffe.Net(model_def, pretrained_model)
  NET.set_phase_test()
  if gpu:
    NET.set_mode_gpu()
  print("Caffe model loaded in {:.3f} s".format(time.time() - t))

  # Configure for input/output data
  IMAGE_DIM = image_dim
  CROPPED_DIM = NET.blobs.values()[0].width
  IMAGE_CENTER = int((IMAGE_DIM - CROPPED_DIM) / 2)

    # Load the data set mean file
  IMAGE_MEAN = np.load(image_mean_file)

  CROPPED_IMAGE_MEAN = IMAGE_MEAN[IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  IMAGE_CENTER:IMAGE_CENTER + CROPPED_DIM,
                                  :]
  BATCH_SIZE = NET.blobs.values()[0].num  # network batch size
  NUM_OUTPUT = NET.blobs.values()[-1].channels  # number of output classes


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Required arguments: input and output.
  parser.add_argument(
    "input_file",
    help="Input txt/csv filename. If .txt, must be list of filenames.\
    If .csv, must be comma-separated file with header\
    'filename, xmin, ymin, xmax, ymax'"
  )
  parser.add_argument(
    "output_file",
    help="Output h5/csv filename. Format depends on extension."
  )

  # Optional arguments.
  parser.add_argument(
    "--model_def",
    default="../../../examples/imagenet/imagenet_deploy.prototxt",
    help="Model definition file."
  )
  parser.add_argument(
    "--pretrained_model",
    default="../../../examples/imagenet/caffe_reference_imagenet_model",
    help="Trained model weights file."
  )
  parser.add_argument(
    "--gpu",
    default=False,
    help="Switch for gpu computation."
  )
  parser.add_argument(
    "--crop_mode",
    default="center_only",
    choices=CROP_MODES,
    help="Image crop mode"
  )
  parser.add_argument(
    "--images_dim",
    default=256,
    help="Canonical dimension of (square) images."
  )
  parser.add_argument(
    "--images_mean_file",
    default=os.path.join(
      os.path.dirname(__file__), '../imagenet/ilsvrc_2012_mean.npy'),
    help="Data set image mean (numpy array).")

  args = parser.parse_args()

  # Configure network, input, output.
  config(args.model_def, args.pretrained_model, args.gpu, args.images_dim,
         args.images_mean_file)

  # Load input.
  t = time.time()
  print('Loading input and assembling batches...')
  if args.input_file.lower().endswith('txt'):
    with open(args.input_file) as f:
      inputs = [_.strip() for _ in f.readlines()]
  elif args.input_file.lower().endswith('csv'):
    inputs = pd.read_csv(args.input_file, sep=',', dtype={'filename': str})
    inputs.set_index('filename', inplace=True)
  else:
    raise Exception("Uknown input file type: not in txt or csv")

  # Assemble into batches
  image_batches = assemble_batches(inputs, args.crop_mode)
  print('{} batches assembled in {:.3f} s'.format(len(image_batches),
                                                  time.time() - t))

  # Process the batches.
  t = time.time()
  print 'Processing {} files in {} batches'.format(len(inputs),
                                                   len(image_batches))
  dfs_with_feats = []
  for i in range(len(image_batches)):
    if i % 10 == 0:
      print('...on batch {}/{}, elapsed time: {:.3f} s'.format(
        i, len(image_batches), time.time() - t))
    dfs_with_feats.append(compute_feats(image_batches[i]))

  # Concatenate, droppping the padding rows.
  df = pd.concat(dfs_with_feats).dropna(subset=['filename'])
  df.set_index('filename', inplace=True)
  print("Processing complete after {:.3f} s.".format(time.time() - t))

  # Label coordinates
  coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
  df[coord_cols] = pd.DataFrame(
    data=np.vstack(df['window']), index=df.index, columns=coord_cols)
  del(df['window'])

  # Write out the results.
  t = time.time()
  if args.output_file.lower().endswith('csv'):
    # enumerate the class probabilities
    class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
    df[class_cols] = pd.DataFrame(
      data=np.vstack(df['feat']), index=df.index, columns=class_cols)
    df.to_csv(args.output_file, cols=coord_cols + class_cols)
  else:
    df.to_hdf(args.output_file, 'df', mode='w')
  print("Done. Saving to {} took {:.3f} s.".format(
    args.output_file, time.time() - t))

  sys.exit()
