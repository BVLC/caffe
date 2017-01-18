---
title: Inner Product / Fully Connected Layer
---

# Inner Product / Fully Connected Layer

* Layer type: `InnerProduct`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1InnerProductLayer.html)
* Header: [`./include/caffe/layers/inner_product_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/inner_product_layer.hpp)
* CPU implementation: [`./src/caffe/layers/inner_product_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/inner_product_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cu)

* Input
    - `n * c_i * h_i * w_i`
* Output
    - `n * c_o * 1 * 1`
* Sample

      layer {
        name: "fc8"
        type: "InnerProduct"
        # learning rate and decay multipliers for the weights
        param { lr_mult: 1 decay_mult: 1 }
        # learning rate and decay multipliers for the biases
        param { lr_mult: 2 decay_mult: 0 }
        inner_product_param {
          num_output: 1000
          weight_filler {
            type: "gaussian"
            std: 0.01
          }
          bias_filler {
            type: "constant"
            value: 0
          }
        }
        bottom: "fc7"
        top: "fc8"
      }

The `InnerProduct` layer (also usually referred to as the fully connected layer) treats the input as a simple vector and produces an output in the form of a single vector (with the blob's height and width set to 1).


## Parameters

* Parameters (`InnerProductParameter inner_product_param`)
    - Required
        - `num_output` (`c_o`): the number of filters
    - Strongly recommended
        - `weight_filler` [default `type: 'constant' value: 0`]
    - Optional
        - `bias_filler` [default `type: 'constant' value: 0`]
        - `bias_term` [default `true`]: specifies whether to learn and apply a set of additive biases to the filter outputs
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/InnerProductParameter.txt %}
{% endhighlight %}
 
