---
title: Softmax Layer
---

# Softmax Layer

* Layer type: `Softmax`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SoftmaxLayer.html)
* Header: [`./include/caffe/layers/softmax_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/softmax_layer.hpp)
* CPU implementation: [`./src/caffe/layers/softmax_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/softmax_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_layer.cu)

## Parameters

* Parameters (`SoftmaxParameter softmax_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/SoftmaxParameter.txt %}
{% endhighlight %}

## See also

* [Softmax loss layer](softmaxwithloss.html)
