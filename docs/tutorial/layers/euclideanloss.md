---
title: Euclidean Loss Layer
---
# Sum-of-Squares / Euclidean Loss Layer

* Layer type: `EuclideanLoss`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1EuclideanLossLayer.html)
* Header: [`./include/caffe/layers/euclidean_loss_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/euclidean_loss_layer.hpp)
* CPU implementation: [`./src/caffe/layers/euclidean_loss_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/euclidean_loss_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cu)

The Euclidean loss layer computes the sum of squares of differences of its two inputs, $$\frac 1 {2N} \sum_{i=1}^N \| x^1_i - x^2_i \|_2^2$$.

## Parameters

Does not take any parameters.
