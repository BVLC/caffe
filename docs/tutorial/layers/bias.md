---
title: Bias Layer
---

# Bias Layer

* Layer type: `Bias`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1BiasLayer.html)
* Header: [`./include/caffe/layers/bias_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/bias_layer.hpp)
* CPU implementation: [`./src/caffe/layers/bias_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/bias_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/bias_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/bias_layer.cu)

## Parameters
* Parameters (`BiasParameter bias_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/BiasParameter.txt %}
{% endhighlight %}
