---
---
# Layers

To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt).

Caffe layers and their parameters are defined in the protocol buffer definitions for the project in [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto). The latest definitions are in the [dev caffe.proto](https://github.com/BVLC/caffe/blob/dev/src/caffe/proto/caffe.proto).

TODO complete list of layers linking to headings

### Vision Layers

* Header: `./include/caffe/vision_layers.hpp`

Vision layers usually take *images* as input and produce other *images* as output.
A typical "image" in the real-world may have one color channel ($c = 1$), as in a grayscale image, or three color channels ($c = 3$) as in an RGB (red, green, blue) image.
But in this context, the distinguishing characteristic of an image is its spatial structure: usually an image has some non-trivial height $h > 1$ and width $w > 1$.
This 2D geometry naturally lends itself to certain decisions about how to process the input.
In particular, most of the vision layers work by applying a particular operation to some region of the input to produce a corresponding region of the output.
In contrast, other layers (with few exceptions) ignore the spatial structure of the input, treating it as "one big vector" with dimension $$ c h w $$.


#### Convolution

* LayerType: `CONVOLUTION`
* CPU implementation: `./src/caffe/layers/convolution_layer.cpp`
* CUDA GPU implementation: `./src/caffe/layers/convolution_layer.cu`
* Options (`ConvolutionParameter convolution_param`)
    - Required | `num_output` ($c_o$), the number of filters
    - Required: `kernel_size` or (`kernel_h`, `kernel_w`), specifies height & width of each filter
    - Strongly recommended (default `type: 'constant' value: 0`): `weight_filler`
    - Optional (default `true`): `bias_term`, specifies whether to learn and apply a set of additive biases to the filter outputs
    - Optional (default 0): `pad` or (`pad_h`, `pad_w`), specifies the number of pixels to (implicitly) add to each side of the input
    - Optional (default 1): `stride` or (`stride_h`, `stride_w`), specifies the intervals at which to apply the filters to the input
    - Optional (default 1): `group` ($g$) if $>1$, restricts the connectivity of each filter to a subset of the input.  In particular, the input to the $i^{th}$ group of $n_f / g$ filters is the $i^{th}$ group of $c_i / g$ input channels.
* Input
    - $n \times c_i \times h_i \times w_i$ (repeated $K \ge 1$ times)
* Output
    - $n \times c_o \times h_o \times w_o$ (repeated $K$ times)
* Sample (as seen in `./examples/imagenet/imagenet_train_val.prototxt`)

        layers {
          name: "conv1"
          type: CONVOLUTION
          bottom: "data"
          top: "conv1"
          blobs_lr: 1          # learning rate multiplier for the filters
          blobs_lr: 2          # learning rate multiplier for the biases
          weight_decay: 1      # weight decay multiplier for the filters
          weight_decay: 0      # weight decay multiplier for the biases
          convolution_param {
            num_output: 96     # learn 96 filters
            kernel_size: 11    # each filter is 11x11
            stride: 4          # step 4 pixels between each filter application
            weight_filler {
              type: "gaussian" # initialize the filters from a Gaussian
              std: 0.01        # distribution with stdev 0.01 (default mean: 0)
            }
            bias_filler {
              type: "constant" # initialize the biases to zero (0)
              value: 0
            }
          }
        }

The `CONVOLUTION` layer convolves the input image with a set of learnable filters, each producing one feature map in the output image.

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
