---
title: Batch Norm Layer
---

# Batch Norm Layer

* Layer type: `BatchNorm`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1BatchNormLayer.html)
* Header: [`./include/caffe/layers/batch_norm_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/batch_norm_layer.hpp)
* CPU implementation: [`./src/caffe/layers/batch_norm_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/batch_norm_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cu)

## Parameters

* Parameters (`BatchNormParameter batch_norm_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/BatchNormParameter.txt %}
{% endhighlight %}
