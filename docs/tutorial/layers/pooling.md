---
title: Pooling Layer
---
# Pooling

* Layer type: `Pooling`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1PoolingLayer.html)
* Header: [`./include/caffe/layers/pooling_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/pooling_layer.hpp)
* CPU implementation: [`./src/caffe/layers/pooling_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/pooling_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu)

* Input
    - `n * c * h_i * w_i`
* Output
    - `n * c * h_o * w_o`, where h_o and w_o are computed in the same way as convolution.

## Parameters

* Parameters (`PoolingParameter pooling_param`)
    - Required
        - `kernel_size` (or `kernel_h` and `kernel_w`): specifies height and width of each filter
    - Optional
        - `pool` [default MAX]: the pooling method. Currently MAX, AVE, or STOCHASTIC
        - `pad` (or `pad_h` and `pad_w`) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input
        - `stride` (or `stride_h` and `stride_w`) [default 1]: specifies the intervals at which to apply the filters to the input


* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/PoolingParameter.txt %}
{% endhighlight %}

## Sample
* Sample (as seen in [`./models/bvlc_reference_caffenet/train_val.prototxt`](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/train_val.prototxt))

      layer {
        name: "pool1"
        type: "Pooling"
        bottom: "conv1"
        top: "pool1"
        pooling_param {
          pool: MAX
          kernel_size: 3 # pool over a 3x3 region
          stride: 2      # step two pixels (in the bottom blob) between pooling regions
        }
      }
