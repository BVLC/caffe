---
title: HDF5 Output Layer
---

# HDF5 Output Layer

* Layer type: `HDF5Output`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1HDF5OutputLayer.html)
* Header: [`./include/caffe/layers/hdf5_output_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/hdf5_output_layer.hpp)
* CPU implementation: [`./src/caffe/layers/hdf5_output_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/hdf5_output_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/hdf5_output_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/hdf5_output_layer.cu)

The HDF5 output layer performs the opposite function of the other layers in this section: it writes its input blobs to disk.

## Parameters

* Parameters (`HDF5OutputParameter hdf5_output_param`)
    - Required
        - `file_name`: name of file to write to

* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/HDF5OutputParameter.txt %}
{% endhighlight %}
