---
title: Parameter Layer
---

# Parameter Layer

* Layer type: `Parameter`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ParameterLayer.html)
* Header: [`./include/caffe/layers/parameter_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/parameter_layer.hpp)
* CPU implementation: [`./src/caffe/layers/parameter_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/parameter_layer.cpp)

See [https://github.com/BVLC/caffe/pull/2079](https://github.com/BVLC/caffe/pull/2079).

## Parameters

* Parameters (`ParameterParameter parameter_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ParameterParameter.txt %}
{% endhighlight %}
