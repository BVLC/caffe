#!/usr/bin/env python
"""
Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows ideas in
    Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
    Rich feature hierarchies for accurate object detection and semantic
    segmentation.
    http://arxiv.org/abs/1311.2524

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO
- R-CNN crop mode / crop with context.
- Bundle with R-CNN model for example.
"""
import numpy as np
import os

import caffe


class Detector(caffe.Net):
    """
    Detector extends Net for windowed detection by a list of crops or
    selective search proposals.
    """
    def __init__(self, model_file, pretrained_file, gpu=False, mean_file=None,
                 input_scale=None, channel_swap=None):
        """
        Take
        gpu, mean_file, input_scale, channel_swap: convenience params for
            setting mode, mean, input scale, and channel order.
        """
        caffe.Net.__init__(self, model_file, pretrained_file)
        self.set_phase_test()

        if gpu:
            self.set_mode_gpu()
        else:
            self.set_mode_cpu()

        if mean_file:
            self.set_mean(self.inputs[0], mean_file)
        if input_scale:
            self.set_input_scale(self.inputs[0], input_scale)
        if channel_swap:
            self.set_channel_swap(self.inputs[0], channel_swap)


    def detect_windows(self, images_windows):
        """
        Do windowed detection over given images and windows. Windows are
        extracted then warped to the input dimensions of the net.

        Take
        images_windows: (image filename, window list) iterable.

        Give
        detections: list of {filename: image filename, window: crop coordinates,
            predictions: prediction vector} dicts.
        """
        # Extract windows.
        window_inputs = []
        for image_fname, windows in images_windows:
            image = caffe.io.load_image(image_fname).astype(np.float32)
            for window in windows:
                window_inputs.append(image[window[0]:window[2],
                                           window[1]:window[3]])

        # Run through the net (warping windows to input dimensions).
        caffe_in = np.asarray([self.preprocess(self.inputs[0], window_in)
                    for window_in in window_inputs])
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]].squeeze(axis=(2,3))

        # Package predictions with images and windows.
        detections = []
        ix = 0
        for image_fname, windows in images_windows:
            for window in windows:
                detections.append({
                    'window': window,
                    'prediction': predictions[ix],
                    'filename': image_fname
                })
                ix += 1
        return detections


    def detect_selective_search(self, image_fnames):
        """
        Do windowed detection over Selective Search proposals by extracting
        the crop and warping to the input dimensions of the net.

        Take
        image_fnames: list

        Give
        detections: list of {filename: image filename, window: crop coordinates,
            predictions: prediction vector} dicts.
        """
        import selective_search_ijcv_with_python as selective_search
        # Make absolute paths so MATLAB can find the files.
        image_fnames = [os.path.abspath(f) for f in image_fnames]
        windows_list = selective_search.get_windows(image_fnames)
        # Run windowed detection on the selective search list.
        return self.detect_windows(zip(image_fnames, windows_list))
