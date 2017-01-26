---
title: Local Response Normalization (LRN)
---

# Local Response Normalization (LRN)

* Layer type: `LRN`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1LRNLayer.html)
* Header: [`./include/caffe/layers/lrn_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/lrn_layer.hpp)
* CPU Implementation: [`./src/caffe/layers/lrn_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lrn_layer.cpp)
* CUDA GPU Implementation: [`./src/caffe/layers/lrn_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lrn_layer.cu)
* Parameters (`LRNParameter lrn_param`)
    - Optional
        - `local_size` [default 5]: the number of channels to sum over (for cross channel LRN) or the side length of the square region to sum over (for within channel LRN)
        - `alpha` [default 1]: the scaling parameter (see below)
        - `beta` [default 5]: the exponent (see below)
        - `norm_region` [default `ACROSS_CHANNELS`]: whether to sum over adjacent channels (`ACROSS_CHANNELS`) or nearby spatial locaitons (`WITHIN_CHANNEL`)

The local response normalization layer performs a kind of "lateral inhibition" by normalizing over local input regions. In `ACROSS_CHANNELS` mode, the local regions extend across nearby channels, but have no spatial extent (i.e., they have shape `local_size x 1 x 1`). In `WITHIN_CHANNEL` mode, the local regions extend spatially, but are in separate channels (i.e., they have shape `1 x local_size x local_size`). Each input value is divided by $$(1 + (\alpha/n) \sum_i x_i^2)^\beta$$, where $$n$$ is the size of each local region, and the sum is taken over the region centered at that value (zero padding is added where necessary).

## Parameters

* Parameters (`LRNParameter lrn_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/LRNParameter.txt %}
{% endhighlight %}
