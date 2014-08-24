---
layout: default
---
# Layers

To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt).

Caffe layers and their parameters are defined in the protocol buffer definitions for the project in [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto). The latest definitions are in the [dev caffe.proto](https://github.com/BVLC/caffe/blob/dev/src/caffe/proto/caffe.proto).

TODO complete list of layers linking to headings

### Vision Layers

#### Convolution

`CONVOLUTION`

#### Pooling**

`POOLING`

#### Local Response Normalization

`LRN`

#### im2col

`IM2COL` is a helper for doing the image-to-column transformation that you most likely do not need to know about.

### Loss Layers

Loss drives learning by comparing an output to a target and assigning cost to minimize. The loss itself is computed by the forward pass and the gradient w.r.t. to the loss is computed by the backward pass.

#### Softmax

`SOFTMAX_LOSS`

#### Sum-of-Squares / Euclidean

`EUCLIDEAN_LOSS`

#### Hinge / Margin

`HINGE_LOSS`

#### Sigmoid Cross-Entropy

`SIGMOID_CROSS_ENTROPY_LOSS`

#### Infogain

`INFOGAIN_LOSS`

#### Accuracy and Top-k

`ACCURACY` scores the output as the accuracy of output with respect to target -- it is not actually a loss and has no backward step.

### Activation / Neuron Layers

#### ReLU / Rectified-Linear and Leaky ReLU

`RELU`

#### Sigmoid

`SIGMOID`

#### TanH / Hyperbolic Tangent

`TANH`

#### Absolute Value

`ABSVAL`

#### Power

`POWER`

#### BNLL

`BNLL`

### Data Layers

#### Database

`DATA`

#### In-Memory

`MEMORY_DATA`

#### HDF5 Input

`HDF5_DATA`

#### HDF5 Output

`HDF5_OUTPUT`

#### Images

`IMAGE_DATA`

#### Windows

`WINDOW_DATA`

#### Dummy

`DUMMY_DATA` is for development and debugging. See `DummyDataParameter`.

### Common Layers

#### Inner Product

`INNER_PRODUCT`

#### Splitting

`SPLIT`

#### Flattening

`FLATTEN`

#### Concatenation

`CONCAT`

#### Slicing

`SLICE`

#### Elementwise Operations

`ELTWISE`

#### Argmax

`ARGMAX`

#### Softmax

`SOFTMAX`

#### Mean-Variance Normalization

`MVN`
