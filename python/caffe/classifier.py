#!/usr/bin/env python
"""
Classifier is an image classifier specialization of Net.
"""

import numpy as np

import caffe


class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 gpu=False, mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        """
        Take
        image_dims: dimensions to scale input for cropping/sampling.
            Default is to scale to net input size for whole-image crop.
        gpu, mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        """
        caffe.Net.__init__(self, model_file, pretrained_file)
        self.set_phase_test()

        if gpu:
            self.set_mode_gpu()
        else:
            self.set_mode_cpu()

        if mean is not None:
            self.set_mean(self.inputs[0], mean)
        if input_scale is not None:
            self.set_input_scale(self.inputs[0], input_scale)
        if raw_scale is not None:
            self.set_raw_scale(self.inputs[0], raw_scale)
        if channel_swap is not None:
            self.set_channel_swap(self.inputs[0], channel_swap)

        self.crop_dims = np.array(self.blobs[self.inputs[0]].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims


    def predict(self, inputs, oversample=True):
        """
        Predict classification probabilities of inputs.

        Take
        inputs: iterable of (H x W x K) input ndarrays.
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.

        Give
        predictions: (N x C) ndarray of class probabilities
                     for N images and C classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]].squeeze(axis=(2,3))

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions
