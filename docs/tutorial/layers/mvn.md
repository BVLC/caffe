---
title: Mean-Variance Normalization (MVN) Layer
---

# Mean-Variance Normalization (MVN) Layer

* Layer type: `MVN`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1MVNLayer.html)
* Header: [`./include/caffe/layers/mvn_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/mvn_layer.hpp)
* CPU implementation: [`./src/caffe/layers/mvn_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/mvn_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/mvn_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/mvn_layer.cu)

## Parameters

* Parameters (`MVNParameter mvn_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/MVNParameter.txt %}
{% endhighlight %}
