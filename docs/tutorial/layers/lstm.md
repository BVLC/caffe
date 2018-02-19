---
title: LSTM Layer
---

# LSTM Layer

* Layer type: `LSTM`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1LSTMLayer.html)
* Header: [`./include/caffe/layers/lstm_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/lstm_layer.hpp)
* CPU implementation: [`./src/caffe/layers/lstm_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_layer.cpp)
* CPU implementation (helper): [`./src/caffe/layers/lstm_unit_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_unit_layer.cpp)
* CUDA GPU implementation (helper): [`./src/caffe/layers/lstm_unit_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lstm_unit_layer.cu)

## Parameters

* Parameters (`Parameter recurrent_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/RecurrentParameter.txt %}
{% endhighlight %}
