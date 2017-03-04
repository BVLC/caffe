---
title: Layer Catalogue
---

# Layers

To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt).

Caffe layers and their parameters are defined in the protocol buffer definitions for the project in [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto).

## Data Layers

Data enters Caffe through data layers: they lie at the bottom of nets. Data can come from efficient databases (LevelDB or LMDB), directly from memory, or, when efficiency is not critical, from files on disk in HDF5 or common image formats.

Common input preprocessing (mean subtraction, scaling, random cropping, and mirroring) is available by specifying `TransformationParameter`s by some of the layers.
The [bias](layers/bias.html), [scale](layers/scale.html), and [crop](layers/crop.html) layers can be helpful with transforming the inputs, when `TransformationParameter` isn't available.

Layers:

* [Image Data](layers/imagedata.html) - read raw images.
* [Database](layers/data.html) - read data from LEVELDB or LMDB.
* [HDF5 Input](layers/hdf5data.html) - read HDF5 data, allows data of arbitrary dimensions.
* [HDF5 Output](layers/hdf5output.html) - write data as HDF5.
* [Input](layers/input.html) - typically used for networks that are being deployed.
* [Window Data](layers/windowdata.html) - read window data file.
* [Memory Data](layers/memorydata.html) - read data directly from memory.
* [Dummy Data](layers/dummydata.html) - for static data and debugging.

Note that the [Python](layers/python.html) Layer can be useful for create custom data layers.

## Vision Layers

Vision layers usually take *images* as input and produce other *images* as output, although they can take data of other types and dimensions.
A typical "image" in the real-world may have one color channel ($$c = 1$$), as in a grayscale image, or three color channels ($$c = 3$$) as in an RGB (red, green, blue) image.
But in this context, the distinguishing characteristic of an image is its spatial structure: usually an image has some non-trivial height $$h > 1$$ and width $$w > 1$$.
This 2D geometry naturally lends itself to certain decisions about how to process the input.
In particular, most of the vision layers work by applying a particular operation to some region of the input to produce a corresponding region of the output.
In contrast, other layers (with few exceptions) ignore the spatial structure of the input, effectively treating it as "one big vector" with dimension $$chw$$.

Layers:

* [Convolution Layer](layers/convolution.html) - convolves the input image with a set of learnable filters, each producing one feature map in the output image.
* [Pooling Layer](layers/pooling.html) - max, average, or stochastic pooling.
* [Spatial Pyramid Pooling (SPP)](layers/spp.html)
* [Crop](layers/crop.html) - perform cropping transformation.
* [Deconvolution Layer](layers/deconvolution.html) - transposed convolution.

* [Im2Col](layers/im2col.html) - relic helper layer that is not used much anymore.

## Recurrent Layers

Layers:

* [Recurrent](layers/recurrent.html)
* [RNN](layers/rnn.html)
* [Long-Short Term Memory (LSTM)](layers/lstm.html)

## Common Layers

Layers:

* [Inner Product](layers/innerproduct.html) - fully connected layer.
* [Dropout](layers/dropout.html)
* [Embed](layers/embed.html) - for learning embeddings of one-hot encoded vector (takes index as input).

## Normalization Layers

* [Local Response Normalization (LRN)](layers/lrn.html) - performs a kind of "lateral inhibition" by normalizing over local input regions.
* [Mean Variance Normalization (MVN)](layers/mvn.html) - performs contrast normalization / instance normalization.
* [Batch Normalization](layers/batchnorm.html) - performs normalization over mini-batches.

The [bias](layers/bias.html) and [scale](layers/scale.html) layers can be helpful in combination with normalization.

## Activation / Neuron Layers

In general, activation / Neuron layers are element-wise operators, taking one bottom blob and producing one top blob of the same size. In the layers below, we will ignore the input and out sizes as they are identical:

* Input
    - n * c * h * w
* Output
    - n * c * h * w

Layers:

* [ReLU / Rectified-Linear and Leaky-ReLU](layers/relu.html) - ReLU and Leaky-ReLU rectification.
* [PReLU](layers/prelu.html) - parametric ReLU.
* [ELU](layers/elu.html) - exponential linear rectification.
* [Sigmoid](layers/sigmoid.html)
* [TanH](layers/tanh.html)
* [Absolute Value](layers/abs.html)
* [Power](layers/power.html) - f(x) = (shift + scale * x) ^ power.
* [Exp](layers/exp.html) - f(x) = base ^ (shift + scale * x).
* [Log](layers/log.html) - f(x) = log(x).
* [BNLL](layers/bnll.html) - f(x) = log(1 + exp(x)).
* [Threshold](layers/threshold.html) - performs step function at user defined threshold.
* [Bias](layers/bias.html) - adds a bias to a blob that can either be learned or fixed.
* [Scale](layers/scale.html) - scales a blob by an amount that can either be learned or fixed.

## Utility Layers

Layers:

* [Flatten](layers/flatten.html)
* [Reshape](layers/reshape.html)
* [Batch Reindex](layers/batchreindex.html)

* [Split](layers/split.html)
* [Concat](layers/concat.html)
* [Slicing](layers/slice.html)
* [Eltwise](layers/eltwise.html) - element-wise operations such as product or sum between two blobs.
* [Filter / Mask](layers/filter.html) - mask or select output using last blob.
* [Parameter](layers/parameter.html) - enable parameters to be shared between layers.
* [Reduction](layers/reduction.html) - reduce input blob to scalar blob using operations such as sum or mean.
* [Silence](layers/silence.html) - prevent top-level blobs from being printed during training.

* [ArgMax](layers/argmax.html)
* [Softmax](layers/softmax.html)

* [Python](layers/python.html) - allows custom Python layers.

## Loss Layers

Loss drives learning by comparing an output to a target and assigning cost to minimize. The loss itself is computed by the forward pass and the gradient w.r.t. to the loss is computed by the backward pass.

Layers:

* [Multinomial Logistic Loss](layers/multinomiallogisticloss.html)
* [Infogain Loss](layers/infogainloss.html) - a generalization of MultinomialLogisticLossLayer.
* [Softmax with Loss](layers/softmaxwithloss.html) - computes the multinomial logistic loss of the softmax of its inputs. It's conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.
* [Sum-of-Squares / Euclidean](layers/euclideanloss.html) - computes the sum of squares of differences of its two inputs, $$\frac 1 {2N} \sum_{i=1}^N \| x^1_i - x^2_i \|_2^2$$.
* [Hinge / Margin](layers/hingeloss.html) - The hinge loss layer computes a one-vs-all hinge (L1) or squared hinge loss (L2).
* [Sigmoid Cross-Entropy Loss](layers/sigmoidcrossentropyloss.html) - computes the cross-entropy (logistic) loss, often used for predicting targets interpreted as probabilities.
* [Accuracy / Top-k layer](layers/accuracy.html) - scores the output as an accuracy with respect to target -- it is not actually a loss and has no backward step.
* [Contrastive Loss](layers/contrastiveloss.html)

