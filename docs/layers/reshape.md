---
title: Reshape Layer
---

# Reshape Layer
* Layer type: `Reshape`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ReshapeLayer.html)
* Header: [`./include/caffe/layers/reshape_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/reshape_layer.hpp)
* Implementation: [`./src/caffe/layers/reshape_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/reshape_layer.cpp)

* Input
    - a single blob with arbitrary dimensions
* Output
    - the same blob, with modified dimensions, as specified by `reshape_param`

* Sample

        layer {
          name: "reshape"
          type: "Reshape"
          bottom: "input"
          top: "output"
          reshape_param {
            shape {
              dim: 0  # copy the dimension from below
              dim: 2
              dim: 3
              dim: -1 # infer it from the other dimensions
            }
          }
        }

The `Reshape` layer can be used to change the dimensions of its input, without changing its data. Just like the `Flatten` layer, only the dimensions are changed; no data is copied in the process.

Output dimensions are specified by the `ReshapeParam` proto. Positive numbers are used directly, setting the corresponding dimension of the output blob. In addition, two special values are accepted for any of the target dimension values:

* **0** means "copy the respective dimension of the bottom layer". That is, if the bottom has 2 as its 1st dimension, the top will have 2 as its 1st dimension as well, given `dim: 0` as the 1st target dimension.
* **-1** stands for "infer this from the other dimensions". This behavior is similar to that of -1 in *numpy*'s or `[]` for *MATLAB*'s reshape: this dimension is calculated to keep the overall element count the same as in the bottom layer. At most one -1 can be used in a reshape operation.

As another example, specifying `reshape_param { shape { dim: 0 dim: -1 } }` makes the layer behave in exactly the same way as the `Flatten` layer.
 
## Parameters

* Parameters (`ReshapeParameter reshape_param`)
    - Optional: (also see detailed description below)
        - `shape`
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ReshapeParameter.txt %}
{% endhighlight %}
