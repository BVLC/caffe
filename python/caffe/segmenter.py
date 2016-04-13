#!/usr/bin/env python
"""
Segmenter is an image segmentation specialization of Net.
"""

import numpy as np

import caffe


class Segmenter(caffe.Net):
    """
    Segmenter
    """
    def __init__(self, model_file, pretrained_file,gpu=False):
        """
        """
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        #self.set_phase_test()

        if gpu:
            caffe.set_mode_gpu()#self.set_mode_gpu()
            caffe.set_device(0)#self.set_device(0)
        else:
            caffe.set_mode_cpu()#self.set_mode_cpu()


    def predict(self, inputs):
        """
        Assume that the input is a 500 x 500 image BRG layout with
        correct padding as necessary to make it 500 x 500.
        """

        input_ = np.zeros((len(inputs),
            500, 500, inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = in_

        # Segment
        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = in_.transpose((2, 0, 1))
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]

        return predictions[0].argmax(axis=0).astype(np.uint8)
