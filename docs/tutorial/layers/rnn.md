---
title: RNN Layer
---

# RNN Layer

* Layer type: `RNN`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1RNNLayer.html)
* Header: [`./include/caffe/layers/rnn_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/rnn_layer.hpp)
* CPU implementation: [`./src/caffe/layers/rnn_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/rnn_layer.cpp)

## Parameters

* Parameters (`RecurrentParameter recurrent_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/RecurrentParameter.txt %}
{% endhighlight %}
