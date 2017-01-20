---
title: Infogain Loss Layer
---

# Infogain Loss Layer

* Layer type: `InfogainLoss`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1InfogainLossLayer.html)
* Header: [`./include/caffe/layers/infogain_loss_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/infogain_loss_layer.hpp)
* CPU implementation: [`./src/caffe/layers/infogain_loss_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/infogain_loss_layer.cpp)

A generalization of [MultinomialLogisticLossLayer](multinomiallogisticloss.html) that takes an "information gain" (infogain) matrix specifying the "value" of all label pairs.

Equivalent to the [MultinomialLogisticLossLayer](multinomiallogisticloss.html) if the infogain matrix is the identity.

## Parameters

* Parameters (`Parameter infogain_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/InfogainLossParameter.txt %}
{% endhighlight %}
