---
title: Absolute Value Layer
---

# Absolute Value Layer

* Layer type: `AbsVal`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1AbsValLayer.html)
* Header: [`./include/caffe/layers/absval_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/absval_layer.hpp)
* CPU implementation: [`./src/caffe/layers/absval_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/absval_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/absval_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/absval_layer.cu)

* Sample

      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "AbsVal"
      }

The `AbsVal` layer computes the output as abs(x) for each input element x.
