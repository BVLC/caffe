---
title: Softmax with Loss Layer
---

# Softmax with Loss Layer

* Layer type: `SoftmaxWithLoss`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SoftmaxWithLossLayer.html)
* Header: [`./include/caffe/layers/softmax_loss_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/softmax_loss_layer.hpp)
* CPU implementation: [`./src/caffe/layers/softmax_loss_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_loss_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/softmax_loss_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_loss_layer.cu)

The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It's conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.

## Parameters

* Parameters (`SoftmaxParameter softmax_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/SoftmaxParameter.txt %}
{% endhighlight %}

* Parameters (`LossParameter loss_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/LossParameter.txt %}
{% endhighlight %}

## See also

* [Softmax layer](softmax.html)
