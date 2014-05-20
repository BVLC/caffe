import numpy as np
import skimage.io
import skimage.transform

from caffe.proto import caffe_pb2


def load_image(filename):
    """
    Load an image converting from grayscale or alpha as needed.

    Take
    filename: string

    Give
    image: an image of size (H x W x 3) with RGB channels of type uint8.
    """
    img = skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Take
    im: (H x W x K) ndarray
    new_dims: (height, width) tuple of new dimensions.
    interp_order: interpolation order, default is linear.

    Give
    im: resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    return skimage.transform.resize(im, new_dims, order=interp_order)


def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Take
    image: iterable of (H x W x K) ndarrays
    crop_dims: (height, width) tuple for the crops.

    Give
    crops: (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                            im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def blobproto_to_array(blob, return_diff=False):
  """Convert a blob proto to an array. In default, we will just return the data,
  unless return_diff is True, in which case we will return the diff.
  """
  if return_diff:
    return np.array(blob.diff).reshape(
        blob.num, blob.channels, blob.height, blob.width)
  else:
    return np.array(blob.data).reshape(
        blob.num, blob.channels, blob.height, blob.width)


def array_to_blobproto(arr, diff=None):
  """Converts a 4-dimensional array to blob proto. If diff is given, also
  convert the diff. You need to make sure that arr and diff have the same
  shape, and this function does not do sanity check.
  """
  if arr.ndim != 4:
    raise ValueError('Incorrect array shape.')
  blob = caffe_pb2.BlobProto()
  blob.num, blob.channels, blob.height, blob.width = arr.shape;
  blob.data.extend(arr.astype(float).flat)
  if diff is not None:
    blob.diff.extend(diff.astype(float).flat)
  return blob


def arraylist_to_blobprotovecor_str(arraylist):
  """Converts a list of arrays to a serialized blobprotovec, which could be
  then passed to a network for processing.
  """
  vec = caffe_pb2.BlobProtoVector()
  vec.blobs.extend([array_to_blobproto(arr) for arr in arraylist])
  return vec.SerializeToString()


def blobprotovector_str_to_arraylist(str):
  """Converts a serialized blobprotovec to a list of arrays.
  """
  vec = caffe_pb2.BlobProtoVector()
  vec.ParseFromString(str)
  return [blobproto_to_array(blob) for blob in vec.blobs]


def array_to_datum(arr, label=0):
  """Converts a 3-dimensional array to datum. If the array has dtype uint8,
  the output data will be encoded as a string. Otherwise, the output data
  will be stored in float format.
  """
  if arr.ndim != 3:
    raise ValueError('Incorrect array shape.')
  datum = caffe_pb2.Datum()
  datum.channels, datum.height, datum.width = arr.shape
  if arr.dtype == np.uint8:
    datum.data = arr.tostring()
  else:
    datum.float_data.extend(arr.flat)
  datum.label = label
  return datum


def datum_to_array(datum):
  """Converts a datum to an array. Note that the label is not returned,
  as one can easily get it by calling datum.label.
  """
  if len(datum.data):
    return np.fromstring(datum.data, dtype = np.uint8).reshape(
        datum.channels, datum.height, datum.width)
  else:
    return np.array(datum.float_data).astype(float).reshape(
        datum.channels, datum.height, datum.width)
