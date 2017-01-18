---
title: PReLU Layer
---

# PReLU Layer

* Layer type: `PReLU`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1PReLULayer.html)
* Header: [`./include/caffe/layers/prelu_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/prelu_layer.hpp)
* CPU implementation: [`./src/caffe/layers/prelu_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/prelu_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/prelu_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/prelu_layer.cu)

## Parameters

* Parameters (`PReLUParameter prelu_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/PReLUParameter.txt %}
{% endhighlight %}
