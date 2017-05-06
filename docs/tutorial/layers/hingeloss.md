---
title: Hinge Loss Layer
---

# Hinge (L1, L2) Loss Layer

* Layer type: `HingeLoss`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1HingeLossLayer.html)
* Header: [`./include/caffe/layers/hinge_loss_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/hinge_loss_layer.hpp)
* CPU implementation: [`./src/caffe/layers/hinge_loss_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/hinge_loss_layer.cpp)

## Parameters

* Parameters (`HingeLossParameter hinge_loss_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/HingeLossParameter.txt %}
{% endhighlight %}
