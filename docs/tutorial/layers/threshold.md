---
title: Threshold Layer
---

# Threshold Layer

* Header: [`./include/caffe/layers/threshold_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/threshold_layer.hpp)
* CPU implementation: [`./src/caffe/layers/threshold_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/threshold_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/threshold_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/threshold_layer.cu)

## Parameters

* Parameters (`ThresholdParameter threshold_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ThresholdParameter.txt %}
{% endhighlight %}
