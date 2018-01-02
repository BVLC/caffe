---
title: Dropout Layer
---

# Dropout Layer

* Layer type: `Dropout`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DropoutLayer.html)
* Header: [`./include/caffe/layers/dropout_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/dropout_layer.hpp)
* CPU implementation: [`./src/caffe/layers/dropout_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/dropout_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cu)

## Parameters

* Parameters (`DropoutParameter dropout_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/DropoutParameter.txt %}
{% endhighlight %}
