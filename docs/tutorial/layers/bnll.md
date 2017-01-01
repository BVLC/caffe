---
title: BNLL Layer
---

# BNLL Layer

* Layer type: `BNLL`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1BNLLLayer.html)
* Header: [`./include/caffe/layers/bnll_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/bnll_layer.hpp)
* CPU implementation: [`./src/caffe/layers/bnll_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/bnll_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/bnll_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/bnll_layer.cu)

The `BNLL` (binomial normal log likelihood) layer computes the output as log(1 + exp(x)) for each input element x.

## Parameters
No parameters.

## Sample

      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: BNLL
      }
