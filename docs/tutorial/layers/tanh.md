---
title: TanH Layer
---

# TanH Layer

* Header: [`./include/caffe/layers/tanh_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/tanh_layer.hpp)
* CPU implementation: [`./src/caffe/layers/tanh_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/tanh_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/tanh_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/tanh_layer.cu)

## Parameters

* Parameters (`TanHParameter tanh_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/TanHParameter.txt %}
{% endhighlight %}
