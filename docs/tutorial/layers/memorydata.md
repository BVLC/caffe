---
title: Memory Data Layer
---

# Memory Data Layer

* Layer type: `MemoryData`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1MemoryDataLayer.html)
* Header: [`./include/caffe/layers/memory_data_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/memory_data_layer.hpp)
* CPU implementation: [`./src/caffe/layers/memory_data_layer.cpu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/memory_data_layer.cpu)

The memory data layer reads data directly from memory, without copying it. In order to use it, one must call `MemoryDataLayer::Reset` (from C++) or `Net.set_input_arrays` (from Python) in order to specify a source of contiguous data (as 4D row major array), which is read one batch-sized chunk at a time.

# Parameters

* Parameters (`MemoryDataParameter memory_data_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/MemoryDataParameter.txt %}
{% endhighlight %}

* Parameters
    - Required
        - `batch_size`, `channels`, `height`, `width`: specify the size of input chunks to read from memory
