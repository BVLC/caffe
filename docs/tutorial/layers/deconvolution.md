---
title: Deconvolution Layer
---

# Deconvolution Layer

* Layer type: `Deconvolution`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DeconvolutionLayer.html)
* Header: [`./include/caffe/layers/deconv_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/deconv_layer.hpp)
* CPU implementation: [`./src/caffe/layers/deconv_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/deconv_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cu)

## Parameters

Uses the same parameters as the Convolution layer.

* Parameters (`ConvolutionParameter convolution_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/ConvolutionParameter.txt %}
{% endhighlight %}
