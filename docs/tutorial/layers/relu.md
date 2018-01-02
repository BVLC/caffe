---
title: ReLU / Rectified-Linear and Leaky-ReLU Layer
---

# ReLU / Rectified-Linear and Leaky-ReLU Layer

* Layer type: `ReLU`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ReLULayer.html)
* Header: [`./include/caffe/layers/relu_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/relu_layer.hpp)
* CPU implementation: [`./src/caffe/layers/relu_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/relu_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/relu_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/relu_layer.cu)
* Sample (as seen in [`./models/bvlc_reference_caffenet/train_val.prototxt`](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/train_val.prototxt))

      layer {
        name: "relu1"
        type: "ReLU"
        bottom: "conv1"
        top: "conv1"
      }

Given an input value x, The `ReLU` layer computes the output as x if x > 0 and negative_slope * x if x <= 0. When the negative slope parameter is not set, it is equivalent to the standard ReLU function of taking max(x, 0). It also supports in-place computation, meaning that the bottom and the top blob could be the same to preserve memory consumption.

## Parameters

* Parameters (`ReLUParameter relu_param`)
    - Optional
        - `negative_slope` [default 0]: specifies whether to leak the negative part by multiplying it with the slope value rather than setting it to 0.
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ReLUParameter.txt %}
{% endhighlight %}
