---
title: HDF5 Data Layer
---

# HDF5 Data Layer

* Layer type: `HDF5Data`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1HDF5DataLayer.html)
* Header: [`./include/caffe/layers/hdf5_data_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/hdf5_data_layer.hpp)
* CPU implementation: [`./src/caffe/layers/hdf5_data_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/hdf5_data_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/hdf5_data_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/hdf5_data_layer.cu)

## Parameters

* Parameters (`HDF5DataParameter hdf5_data_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/HDF5DataParameter.txt %}
{% endhighlight %}
