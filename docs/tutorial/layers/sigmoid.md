---
title: Sigmoid Layer
---

# Sigmoid Layer

* Layer type: `Sigmoid`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SigmoidLayer.html)
* Header: [`./include/caffe/layers/sigmoid_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/sigmoid_layer.hpp)
* CPU implementation: [`./src/caffe/layers/sigmoid_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/sigmoid_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/sigmoid_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/sigmoid_layer.cu)
* Example (from [`./examples/mnist/mnist_autoencoder.prototxt`](https://github.com/BVLC/caffe/blob/master/examples/mnist/mnist_autoencoder.prototxt)):

      layer {
        name: "encode1neuron"
        bottom: "encode1"
        top: "encode1neuron"
        type: "Sigmoid"
      }

The `Sigmoid` layer computes `sigmoid(x)` for each element `x` in the bottom blob.

## Parameters

* Parameters (`SigmoidParameter sigmoid_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/SigmoidParameter.txt %}
{% endhighlight %}
