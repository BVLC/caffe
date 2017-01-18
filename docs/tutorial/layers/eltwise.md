---
title: Eltwise Layer
---

# Eltwise Layer

* Layer type: `Eltwise`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1EltwiseLayer.html)
* Header: [`./include/caffe/layers/eltwise_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/eltwise_layer.hpp)
* CPU implementation: [`./src/caffe/layers/eltwise_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/eltwise_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/eltwise_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/eltwise_layer.cu)

## Parameters

* Parameters (`EltwiseParameter eltwise_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/EltwiseParameter.txt %}
{% endhighlight %}
