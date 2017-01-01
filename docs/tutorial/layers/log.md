---
title: Log Layer
---

# Log Layer

* Layer type: `Log`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1LogLayer.html)
* Header: [`./include/caffe/layers/log_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/log_layer.hpp)
* CPU implementation: [`./src/caffe/layers/log_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/log_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/log_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/log_layer.cu)

## Parameters

* Parameters (`Parameter log_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/LogParameter.txt %}
{% endhighlight %}
