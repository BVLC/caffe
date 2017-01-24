---
title: Contrastive Loss Layer
---

# Contrastive Loss Layer

* Layer type: `ContrastiveLoss`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ContrastiveLossLayer.html)
* Header: [`./include/caffe/layers/contrastive_loss_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/contrastive_loss_layer.hpp)
* CPU implementation: [`./src/caffe/layers/contrastive_loss_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/contrastive_loss_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/contrastive_loss_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/contrastive_loss_layer.cu)

## Parameters

* Parameters (`ContrastiveLossParameter contrastive_loss_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/ContrastiveLossParameter.txt %}
{% endhighlight %}
