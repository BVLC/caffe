---
title: Recurrent Layer
---

# Recurrent Layer

* Layer type: `Recurrent`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1RecurrentLayer.html)
* Header: [`./include/caffe/layers/recurrent_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/recurrent_layer.hpp)
* CPU implementation: [`./src/caffe/layers/recurrent_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/recurrent_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/recurrent_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/recurrent_layer.cu)

## Parameters

* Parameters (`RecurrentParameter recurrent_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/RecurrentParameter.txt %}
{% endhighlight %}
