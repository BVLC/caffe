---
title: Concat Layer
---

# Concat Layer

* Layer type: `Concat`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ConcatLayer.html)
* Header: [`./include/caffe/layers/concat_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/concat_layer.hpp)
* CPU implementation: [`./src/caffe/layers/concat_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/concat_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cu)
* Input
    - `n_i * c_i * h * w` for each input blob i from 1 to K.
* Output
    - if `axis = 0`: `(n_1 + n_2 + ... + n_K) * c_1 * h * w`, and all input `c_i` should be the same.
    - if `axis = 1`: `n_1 * (c_1 + c_2 + ... + c_K) * h * w`, and all input `n_i` should be the same.
* Sample

      layer {
        name: "concat"
        bottom: "in1"
        bottom: "in2"
        top: "out"
        type: "Concat"
        concat_param {
          axis: 1
        }
      }

The `Concat` layer is a utility layer that concatenates its multiple input blobs to one single output blob.

## Parameters
* Parameters (`ConcatParameter concat_param`)
    - Optional
        - `axis` [default 1]: 0 for concatenation along num and 1 for channels.
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/ConcatParameter.txt %}
{% endhighlight %}
