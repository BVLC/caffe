---
title: Silence Layer
---

# Silence Layer

* Layer type: `Silence`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SilenceLayer.html)
* Header: [`./include/caffe/layers/silence_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/silence_layer.hpp)
* CPU implementation: [`./src/caffe/layers/silence_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/silence_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/silence_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/silence_layer.cu)

Silences a blob, so that it is not printed.

## Parameters

* Parameters (`SilenceParameter silence_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/BatchNormParameter.txt %}
{% endhighlight %}

