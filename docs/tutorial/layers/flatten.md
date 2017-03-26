---
title: Flatten Layer
---

# Flatten Layer

* Layer type: `Flatten`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1FlattenLayer.html)
* Header: [`./include/caffe/layers/flatten_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/flatten_layer.hpp)
* CPU implementation: [`./src/caffe/layers/flatten_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/flatten_layer.cpp)

The `Flatten` layer is a utility layer that flattens an input of shape `n * c * h * w` to a simple vector output of shape `n * (c*h*w)`.

## Parameters

* Parameters (`FlattenParameter flatten_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/FlattenParameter.txt %}
{% endhighlight %}
