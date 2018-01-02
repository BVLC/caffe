---
title: Exponential Layer
---

# Exponential Layer

* Layer type: `Exp`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ExpLayer.html)
* Header: [`./include/caffe/layers/exp_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/exp_layer.hpp)
* CPU implementation: [`./src/caffe/layers/exp_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/exp_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/exp_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/exp_layer.cu)

## Parameters

* Parameters (`Parameter exp_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ExpParameter.txt %}
{% endhighlight %}

## See also

* [Power layer](power.html)
