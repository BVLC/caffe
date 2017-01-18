---
title: Sigmoid Layer
---

# Sigmoid Layer

* Layer type: `Sigmoid`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SigmoidLayer.html)
* Header: [`./include/caffe/layers/sigmoid_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/sigmoid_layer.hpp)
* CPU implementation: [`./src/caffe/layers/sigmoid_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/sigmoid_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/sigmoid_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/sigmoid_layer.cu)

## Parameters

* Parameters (`SigmoidParameter sigmoid_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/SigmoidParameter.txt %}
{% endhighlight %}
