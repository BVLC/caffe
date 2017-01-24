---
title: Python Layer
---

# Python Layer

* Layer type: `Python`
* [Doxygen Documentation](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1PythonLayer.html)
* Header: [`./include/caffe/layers/python_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/python_layer.hpp)

The Python layer allows users to add customized layers without modifying the Caffe core code.

## Parameters

* Parameters (`PythonParameter python_param`)
* From [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto):

{% highlight Protobuf %}
{% include proto/PythonParameter.txt %}
{% endhighlight %}

## Examples and tutorials

* Simple Euclidean loss example
** [Python code](https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pyloss.py)
** [Prototxt](https://github.com/BVLC/caffe/blob/master/examples/pycaffe/linreg.prototxt)
* [Tutorial for writing Python layers with DIGITS](https://github.com/NVIDIA/DIGITS/tree/master/examples/python-layer)
