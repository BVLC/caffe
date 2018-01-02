---
title: Database Layer
---

# Database Layer

* Layer type: `Data`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DataLayer.html)
* Header: [`./include/caffe/layers/data_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/data_layer.hpp)
* CPU implementation: [`./src/caffe/layers/data_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/data_layer.cpp)


## Parameters

* Parameters (`DataParameter data_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)):

{% highlight Protobuf %}
{% include proto/DataParameter.txt %}
{% endhighlight %}

* Parameters
    - Required
        - `source`: the name of the directory containing the database
        - `batch_size`: the number of inputs to process at one time
    - Optional
        - `rand_skip`: skip up to this number of inputs at the beginning; useful for asynchronous sgd
        - `backend` [default `LEVELDB`]: choose whether to use a `LEVELDB` or `LMDB`

