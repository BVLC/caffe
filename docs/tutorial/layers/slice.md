---
title: Slice Layer
---

# Slice Layer

* Layer type: `Slice`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1SliceLayer.html)
* Header: [`./include/caffe/layers/slice_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/slice_layer.hpp)
* CPU implementation: [`./src/caffe/layers/slice_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/slice_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/slice_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/slice_layer.cu)

The `Slice` layer is a utility layer that slices an input layer to multiple output layers along a given dimension (currently num or channel only) with given slice indices.

* Sample

      layer {
        name: "slicer_label"
        type: "Slice"
        bottom: "label"
        ## Example of label with a shape N x 3 x 1 x 1
        top: "label1"
        top: "label2"
        top: "label3"
        slice_param {
          axis: 1
          slice_point: 1
          slice_point: 2
        }
      }

`axis` indicates the target axis; `slice_point` indicates indexes in the selected dimension (the number of indices must be equal to the number of top blobs minus one).

## Parameters

* Parameters (`SliceParameter slice_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/SliceParameter.txt %}
{% endhighlight %}

