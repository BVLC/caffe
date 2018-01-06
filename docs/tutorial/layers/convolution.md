---
title: Convolution Layer
---

# Convolution Layer

* Layer type: `Convolution`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ConvolutionLayer.html)
* Header: [`./include/caffe/layers/conv_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/conv_layer.hpp)
* CPU implementation: [`./src/caffe/layers/conv_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/conv_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
* Input
    - `n * c_i * h_i * w_i`
* Output
    - `n * c_o * h_o * w_o`, where `h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1` and `w_o` likewise.

The `Convolution` layer convolves the input image with a set of learnable filters, each producing one feature map in the output image.

## Sample

Sample (as seen in [`./models/bvlc_reference_caffenet/train_val.prototxt`](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/train_val.prototxt)):

      layer {
        name: "conv1"
        type: "Convolution"
        bottom: "data"
        top: "conv1"
        # learning rate and decay multipliers for the filters
        param { lr_mult: 1 decay_mult: 1 }
        # learning rate and decay multipliers for the biases
        param { lr_mult: 2 decay_mult: 0 }
        convolution_param {
          num_output: 96     # learn 96 filters
          kernel_size: 11    # each filter is 11x11
          stride: 4          # step 4 pixels between each filter application
          weight_filler {
            type: "gaussian" # initialize the filters from a Gaussian
            std: 0.01        # distribution with stdev 0.01 (default mean: 0)
          }
          bias_filler {
            type: "constant" # initialize the biases to zero (0)
            value: 0
          }
        }
      }

## Parameters
* Parameters (`ConvolutionParameter convolution_param`)
    - Required
        - `num_output` (`c_o`): the number of filters
        - `kernel_size` (or `kernel_h` and `kernel_w`): specifies height and width of each filter
    - Strongly Recommended
        - `weight_filler` [default `type: 'constant' value: 0`]
    - Optional
        - `bias_term` [default `true`]: specifies whether to learn and apply a set of additive biases to the filter outputs
        - `pad` (or `pad_h` and `pad_w`) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input
        - `stride` (or `stride_h` and `stride_w`) [default 1]: specifies the intervals at which to apply the filters to the input
        - `group` (g) [default 1]: If g > 1, we restrict the connectivity of each filter to a subset of the input. Specifically, the input and output channels are separated into g groups, and the $$i$$th output group channels will be only connected to the $$i$$th input group channels.
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/ConvolutionParameter.txt %}
{% endhighlight %}
