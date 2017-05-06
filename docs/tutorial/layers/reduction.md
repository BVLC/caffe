---
title: Reduction Layer
---

# Reduction Layer

* Layer type: `Reduction`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ReductionLayer.html)
* Header: [`./include/caffe/layers/reduction_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/reduction_layer.hpp)
* CPU implementation: [`./src/caffe/layers/reduction_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/reduction_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/reduction_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/reduction_layer.cu)

## Parameters

* Parameters (`ReductionParameter reduction_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ReductionParameter.txt %}
{% endhighlight %}
