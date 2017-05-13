---
title: Tile Layer
---

# Tile Layer

* Layer type: `Tile`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1TileLayer.html)
* Header: [`./include/caffe/layers/tile_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/tile_layer.hpp)
* CPU implementation: [`./src/caffe/layers/tile_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/tile_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/tile_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/tile_layer.cu)

## Parameters

* Parameters (`TileParameter tile_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/TileParameter.txt %}
{% endhighlight %}
