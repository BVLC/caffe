---
title: Im2col Layer
---

# im2col

* File type: `Im2col`
* Header: [`./include/caffe/layers/im2col_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/im2col_layer.hpp)
* CPU implementation: [`./src/caffe/layers/im2col_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/im2col_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/im2col_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/im2col_layer.cu)

`Im2col` is a helper for doing the image-to-column transformation that you most
likely do not need to know about. This is used in Caffe's original convolution
to do matrix multiplication by laying out all patches into a matrix.


