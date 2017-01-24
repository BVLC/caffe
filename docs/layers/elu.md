---
title: ELU Layer
---

# ELU Layer

* Layer type: `ELU`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ELULayer.html)
* Header: [`./include/caffe/layers/elu_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/elu_layer.hpp)
* CPU implementation: [`./src/caffe/layers/elu_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/elu_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/elu_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/elu_layer.cu)

## References

* Clevert, Djork-Arne, Thomas Unterthiner, and Sepp Hochreiter.
  "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)" [arXiv:1511.07289](https://arxiv.org/abs/1511.07289). (2015).

## Parameters

* Parameters (`ELUParameter elu_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ELUParameter.txt %}
{% endhighlight %}
