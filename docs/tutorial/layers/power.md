---
title: Power Layer
---

# Power Layer

* Layer type: `Power`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1PowerLayer.html)
* Header: [`./include/caffe/layers/power_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/power_layer.hpp)
* CPU implementation: [`./src/caffe/layers/power_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/power_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/power_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/power_layer.cu)

The `Power` layer computes the output as (shift + scale * x) ^ power for each input element x.

## Parameters
* Parameters (`PowerParameter power_param`)
    - Optional
        - `power` [default 1]
        - `scale` [default 1]
        - `shift` [default 0]

* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/PowerParameter.txt %}
{% endhighlight %}
 
 
 
## Sample

      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "Power"
        power_param {
          power: 1
          scale: 1
          shift: 0
        }
      }

## See also

* [Exponential layer](exp.html)
