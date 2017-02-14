import numpy as np
import cv2

try:
    # Python3 will most likely not be able to load protobuf
    from caffe.proto import caffe_pb2
except:
    import sys
    if sys.version_info >= (3, 0):
        print("Failed to include caffe_pb2, things might go wrong!")
    else:
        raise


## proto / datum / ndarray conversion
def blobproto_to_array(blob, return_diff=False):
    """
    Convert a blob proto to an array. In default, we will just return the data,
    unless return_diff is True, in which case we will return the diff.
    """
    # Read the data into an array
    if return_diff:
        data = np.array(blob.diff)
    else:
        data = np.array(blob.data)

    # Reshape the array
    if blob.HasField('num') or blob.HasField('channels') or blob.HasField('height') or blob.HasField('width'):
        # Use legacy 4D shape
        return data.reshape(blob.num, blob.channels, blob.height, blob.width)
    else:
        return data.reshape(blob.shape.dim)

def array_to_blobproto(arr, diff=None):
    """Converts a N-dimensional array to blob proto. If diff is given, also
    convert the diff. You need to make sure that arr and diff have the same
    shape, and this function does not do sanity check.
    """
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    if diff is not None:
        blob.diff.extend(diff.astype(float).flat)
    return blob


def arraylist_to_blobprotovector_str(arraylist):
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


def array_to_datum(arr, label=None):
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
    if label is not None:
        datum.label = label
    return datum


def datum_to_array(datum):
    """Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    """
    if len(datum.data):
        return np.fromstring(datum.data, dtype=np.uint8).reshape(
            datum.channels, datum.height, datum.width)
    else:
        return np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)


## Pre-processing

class Transformer:
    """
    Transform input for feeding into a Net.

    Note: this is mostly for illustrative purposes and it is likely better
    to define your own input preprocessing routine for your needs.

    Parameters
    ----------
    net : a Net for which the input should be prepared
    """
    def __init__(self, inputs):
        self.inputs = inputs
        self.transpose = {}
        self.channel_swap = {}
        self.raw_scale = {}
        self.mean = {}
        self.input_scale = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.format(
                in_, self.inputs))

    def preprocess(self, in_, data):
        """
        Format input for Caffe:
        - convert to single
        - resize to input dimensions (preserving number of channels)
        - transpose dimensions to K x H x W
        - reorder channels (for instance color to BGR)
        - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
        - subtract mean
        - center crop
        - scale feature

        Parameters
        ----------
        in_ : name of input blob to preprocess for
        data : (H' x W' x K) ndarray

        Returns
        -------
        caffe_in : (K x H x W) ndarray for input to a Net
        """
        self.__check_input(in_)
        caffe_in = data.astype(np.float32, copy=False)
        caffe_in_dims = caffe_in.shape[:2]
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        in_dims = self.inputs[in_][2:]
        if mean is not None and mean.shape[1:] > (1,1):
            if caffe_in_dims != mean.shape[1:]:
                caffe_in = resize_image(caffe_in, mean.shape[1:])
                caffe_in_dims = mean.shape[1:]
        else:
            caffe_in = resize_image(caffe_in, in_dims)
            caffe_in_dims = in_dims
        if channel_swap is not None:
            caffe_in = caffe_in[:, :, channel_swap]
        if raw_scale is not None:
            caffe_in *= raw_scale
        if mean is not None:
            caffe_in -= mean.transpose(1,2,0)
        if caffe_in_dims > in_dims:
            caffe_in = caffe_in[(mean.shape[1]-in_dims[0])/2:(mean.shape[1]+in_dims[0])/2,
                                (mean.shape[2]-in_dims[1])/2:(mean.shape[2]+in_dims[1])/2]
        elif caffe_in_dims < in_dims:
            caffe_in = resize_image(caffe_in, in_dims)
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        if input_scale is not None:
            caffe_in *= input_scale
        return caffe_in

    def deprocess(self, in_, data):
        """
        Invert Caffe formatting; see preprocess().
        """
        self.__check_input(in_)
        decaf_in = data.copy().squeeze()
        decaf_in_dims = decaf_in.shape[1:]
        transpose = self.transpose.get(in_)
        channel_swap = self.channel_swap.get(in_)
        raw_scale = self.raw_scale.get(in_)
        mean = self.mean.get(in_)
        input_scale = self.input_scale.get(in_)
        if input_scale is not None:
            decaf_in /= input_scale
        if transpose is not None:
            decaf_in = decaf_in.transpose(np.argsort(transpose))
        if mean is not None:
            if decaf_in_dims < mean.shape[1:]:
                mean = mean.transpose(1,2,0)[(mean.shape[1]-decaf_in_dims[0])/2:(mean.shape[1]+decaf_in_dims[0])/2,
                                             (mean.shape[2]-decaf_in_dims[1])/2:(mean.shape[2]+decaf_in_dims[1])/2]
                decaf_in += mean
            elif decaf_in_dims == mean.shape[1:]:
                mean = mean.transpose(1, 2, 0)
                decaf_in += mean
            else:
                decaf_in = resize_image(decaf_in, mean.shape[1:])
                decaf_in += mean.transpose(1, 2, 0)
                decaf_in = resize_image(decaf_in, decaf_in_dims)
        if raw_scale is not None:
            decaf_in /= raw_scale
        if channel_swap is not None:
            decaf_in = decaf_in[:, :, np.argsort(channel_swap)]

        return decaf_in

    def set_transpose(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to transpose the dimensions
        """
        self.__check_input(in_)
        if len(order) != len(self.inputs[in_]) - 1:
            raise Exception('Transpose order needs to have the same number of '
                            'dimensions as the input.')
        self.transpose[in_] = order

    def set_channel_swap(self, in_, order):
        """
        Set the input channel order for e.g. RGB to BGR conversion
        as needed for the reference ImageNet model.
        N.B. this assumes the channels are the first dimension AFTER transpose.

        Parameters
        ----------
        in_ : which input to assign this channel order
        order : the order to take the channels.
            (2,1,0) maps RGB to BGR for example.
        """
        self.__check_input(in_)
        if len(order) != self.inputs[in_][1]:
            raise Exception('Channel swap needs to have the same number of '
                            'dimensions as the input channels.')
        self.channel_swap[in_] = order

    def set_raw_scale(self, in_, scale):
        """
        Set the scale of raw features s.t. the input blob = input * scale.
        While Python represents images in [0, 1], certain Caffe models
        like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale
        of these models must be 255.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.raw_scale[in_] = scale

    def set_mean(self, in_, mean):
        """
        Set the mean to subtract for centering the data.

        Parameters
        ----------
        in_ : which input to assign this mean.
        mean : mean ndarray (input dimensional or broadcastable)
        """
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
        self.mean[in_] = mean

    def set_input_scale(self, in_, scale):
        """
        Set the scale of preprocessed inputs s.t. the blob = blob * scale.
        N.B. input_scale is done AFTER mean subtraction and other preprocessing
        while raw_scale is done BEFORE.

        Parameters
        ----------
        in_ : which input to assign this scale factor
        scale : scale coefficient
        """
        self.__check_input(in_)
        self.input_scale[in_] = scale


## Image IO

def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    if color:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)[:,:,(2,1,0)]
    else:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = img[:, :, np.newaxis]
    return img.astype(np.float32)/255


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is bilinear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    resized_im = cv2.resize(im, new_dims, interpolation=interp_order)
    return resized_im.astype(np.float32)


def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.

    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
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

def oversample_google(images, crop_dims, interp_order=1):
    """
    Producing 144 crops as described in https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf

    Parameters
    ----------
    images : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops. Must be <= (256,256)
    interp_order : interpolation order, default is bilinear.

    Returns
    -------
    crops : (144*N x H x W x K) ndarray of crops for number of inputs N.
    """

    crops = np.empty((144 * len(images), crop_dims[0], crop_dims[1],
                      images[0].shape[-1]), dtype=np.float32)

    idx = 0
    for img in images:
        size = img.shape[:2]

        for i in [256, 288, 320, 352]:
            ratio = float(i) / min(size)
            new_size = (int(round(size[0] * ratio)), int(round(size[1] * ratio)))
            img_resized = cv2.resize(img, new_size, interpolation=interp_order)

            for j in [0, (max(new_size) - i) / 2, max(new_size) - i]:
                # Landscape
                if new_size[0] > new_size[1]:
                    img_area = img_resized[:, j:j + i]

                # Portrait
                else:
                    img_area = img_resized[j:j + i, :]

                img_area_resized = cv2.resize(img_area, crop_dims, interpolation=interp_order)
                crops[idx] = img_area_resized
                idx += 1
                img_area_resized_mirror = cv2.flip(img_area_resized, 1)
                crops[idx] = img_area_resized_mirror
                idx += 1

                img_area_center = img_area[(i - crop_dims[0]) / 2:(i + crop_dims[0]) / 2, (i - crop_dims[1]) / 2:(i + crop_dims[1]) / 2]
                crops[idx] = img_area_center
                idx += 1
                img_area_center_mirror = cv2.flip(img_area_center, 1)
                crops[idx] = img_area_center_mirror
                idx += 1

                img_area_top_left = img_area[:crop_dims[0], :crop_dims[1]]
                crops[idx] = img_area_top_left
                idx += 1
                img_area_top_left_mirror = cv2.flip(img_area_top_left, 1)
                crops[idx] = img_area_top_left_mirror
                idx += 1

                img_area_top_right = img_area[:crop_dims[0], i - crop_dims[1]:]
                crops[idx] = img_area_top_right
                idx += 1
                img_area_top_right_mirror = cv2.flip(img_area_top_right, 1)
                crops[idx] = img_area_top_right_mirror
                idx += 1

                img_area_bottom_left = img_area[i - crop_dims[0]:, :crop_dims[1]]
                crops[idx] = img_area_bottom_left
                idx += 1
                img_area_bottom_left_mirror = cv2.flip(img_area_bottom_left, 1)
                crops[idx] = img_area_bottom_left_mirror
                idx += 1

                img_area_bottom_right = img_area[i - crop_dims[0]:, i - crop_dims[1]:]
                crops[idx] = img_area_bottom_right
                idx += 1
                img_area_bottom_right_mirror = cv2.flip(img_area_bottom_right, 1)
                crops[idx] = img_area_bottom_right_mirror
                idx += 1

    return crops