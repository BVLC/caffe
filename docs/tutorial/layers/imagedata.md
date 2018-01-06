---
title: ImageData Layer
---

# ImageData Layer

* Layer type: `ImageData`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ImageDataLayer.html)
* Header: [`./include/caffe/layers/image_data_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/image_data_layer.hpp)
* CPU implementation: [`./src/caffe/layers/image_data_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/image_data_layer.cpp)

## Parameters

* Parameters (`ImageDataParameter image_data_parameter`)
    - Required
        - `source`: name of a text file, with each line giving an image filename and label
        - `batch_size`: number of images to batch together
    - Optional
        - `rand_skip`
        - `shuffle` [default false]
        - `new_height`, `new_width`: if provided, resize all images to this size

* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/ImageDataParameter.txt %}
{% endhighlight %}
