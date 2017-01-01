---
title: Crop Layer
---

# Crop Layer

* Layer type: `Crop`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1CropLayer.html)
* Header: [`./include/caffe/layers/crop_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/crop_layer.hpp)
* CPU implementation: [`./src/caffe/layers/crop_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/crop_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/crop_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/crop_layer.cu)

## Parameters

* Parameters (`CropParameter crop_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/CropParameter.txt %}
{% endhighlight %}
