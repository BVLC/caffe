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
"""
import numpy as np
import os

import caffe


class Detector(caffe.Net):
    """
    Detector extends Net for windowed detection by a list of crops or
    selective search proposals.
    """
    def __init__(self, model_file, pretrained_file, gpu=False, mean=None,
                 input_scale=None, raw_scale=None, channel_swap=None,
                 context_pad=None):
        """
        Take
        gpu, mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        context_pad: amount of surrounding context to take s.t. a `context_pad`
            sized border of pixels in the network input image is context, as in
            R-CNN feature extraction.
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

        self.configure_crop(context_pad)


    def detect_windows(self, images_windows):
        """
        Do windowed detection over given images and windows. Windows are
        extracted then warped to the input dimensions of the net.

        Take
        images_windows: (image filename, window list) iterable.
        context_crop: size of context border to crop in pixels.

        Give
        detections: list of {filename: image filename, window: crop coordinates,
            predictions: prediction vector} dicts.
        """
        # Extract windows.
        window_inputs = []
        for image_fname, windows in images_windows:
            image = caffe.io.load_image(image_fname).astype(np.float32)
            for window in windows:
                window_inputs.append(self.crop(image, window))

        # Run through the net (warping windows to input dimensions).
        caffe_in = np.zeros((len(window_inputs), window_inputs[0].shape[2])
                            + self.blobs[self.inputs[0]].data.shape[2:],
                            dtype=np.float32)
        for ix, window_in in enumerate(window_inputs):
            caffe_in[ix] = self.preprocess(self.inputs[0], window_in)
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
        windows_list = selective_search.get_windows(
            image_fnames,
            cmd='selective_search_rcnn'
        )
        # Run windowed detection on the selective search list.
        return self.detect_windows(zip(image_fnames, windows_list))


    def crop(self, im, window):
        """
        Crop a window from the image for detection. Include surrounding context
        according to the `context_pad` configuration.

        Take
        im: H x W x K image ndarray to crop.
        window: bounding box coordinates as ymin, xmin, ymax, xmax.

        Give
        crop: cropped window.
        """
        # Crop window from the image.
        crop = im[window[0]:window[2], window[1]:window[3]]

        if self.context_pad:
            box = window.copy()
            crop_size = self.blobs[self.inputs[0]].width  # assumes square
            scale = crop_size / (1. * crop_size - self.context_pad * 2)
            # Crop a box + surrounding context.
            half_h = (box[2] - box[0] + 1) / 2.
            half_w = (box[3] - box[1] + 1) / 2.
            center = (box[0] + half_h, box[1] + half_w)
            scaled_dims = scale * np.array((-half_h, -half_w, half_h, half_w))
            box = np.round(np.tile(center, 2) + scaled_dims)
            full_h = box[2] - box[0] + 1
            full_w = box[3] - box[1] + 1
            scale_h = crop_size / full_h
            scale_w = crop_size / full_w
            pad_y = round(max(0, -box[0]) * scale_h)  # amount out-of-bounds
            pad_x = round(max(0, -box[1]) * scale_w)

            # Clip box to image dimensions.
            im_h, im_w = im.shape[:2]
            box = np.clip(box, 0., [im_h, im_w, im_h, im_w])
            clip_h = box[2] - box[0] + 1
            clip_w = box[3] - box[1] + 1
            assert(clip_h > 0 and clip_w > 0)
            crop_h = round(clip_h * scale_h)
            crop_w = round(clip_w * scale_w)
            if pad_y + crop_h > crop_size:
                crop_h = crop_size - pad_y
            if pad_x + crop_w > crop_size:
                crop_w = crop_size - pad_x

            # collect with context padding and place in input
            # with mean padding
            context_crop = im[box[0]:box[2], box[1]:box[3]]
            context_crop = caffe.io.resize_image(context_crop, (crop_h, crop_w))
            crop = self.crop_mean.copy()
            crop[pad_y:(pad_y + crop_h), pad_x:(pad_x + crop_w)] = context_crop

        return crop


    def configure_crop(self, context_pad):
        """
        Configure amount of context for cropping.
        If context is included, make the special input mean for context padding.

        Take
        context_pad: amount of context for cropping.
        """
        self.context_pad = context_pad
        if self.context_pad:
            raw_scale = self.raw_scale.get(self.inputs[0])
            channel_order = self.channel_swap.get(self.inputs[0])
            # Padding context crops needs the mean in unprocessed input space.
            mean = self.mean.get(self.inputs[0])
            if mean is not None:
                crop_mean = mean.copy().transpose((1,2,0))
                if channel_order is not None:
                    channel_order_inverse = [channel_order.index(i)
                                            for i in range(crop_mean.shape[2])]
                    crop_mean = crop_mean[:,:, channel_order_inverse]
                if raw_scale is not None:
                    crop_mean /= raw_scale
                self.crop_mean = crop_mean
            else:
                self.crop_mean = np.zeros(self.blobs[self.inputs[0]].data.shape,
                                          dtype=np.float32)
