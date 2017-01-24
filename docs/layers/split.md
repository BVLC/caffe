---
title: Split Layer
---

# Split Layer

* Layer type: `Split`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SplitLayer.html)
* Header: [`./include/caffe/layers/split_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/split_layer.hpp)
* CPU implementation: [`./src/caffe/layers/split_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/split_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/split_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/split_layer.cu)

The `Split` layer is a utility layer that splits an input blob to multiple output blobs. This is used when a blob is fed into multiple output layers.

## Parameters

Does not take any parameters.
