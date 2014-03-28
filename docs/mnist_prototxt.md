---
layout: default
title: Caffe
---

Define the MNIST Network
=========================

This page explains the prototxt file `lenet_train.prototxt` used in the MNIST demo. We assume that you are familiar with [Google Protobuf](https://developers.google.com/protocol-buffers/docs/overview), and assume that you have read the protobuf definitions used by Caffe, which can be found at [src/caffe/proto/caffe.proto](https://github.com/Yangqing/caffe/blob/master/src/caffe/proto/caffe.proto).

Specifically, we will write a `caffe::NetParameter` (or in python, `caffe.proto.caffe_pb2.NetParameter`) protubuf. We will start by giving the network a name:

    name: "LeNet"

Writing the Data Layer
----------------------
Currently, we will read the MNIST data from the leveldb we created earlier in the demo. This is defined by a data layer:

    layers {
      name: "mnist"
      type: DATA
      data_param {
        source: "mnist-train-leveldb"
        batch_size: 64
        scale: 0.00390625
      }
      top: "data"
      top: "label"
    }

Specifically, this layer has name `mnist`, type `data`, and it reads the data from the given leveldb source. We will use a batch size of 64, and scale the incoming pixels so that they are in the range \[0,1\). Why 0.00390625? It is 1 divided by 256. And finally, this layer produces two blobs, one is the `data` blob, and one is the `label` blob.

Writing the Convolution Layer
--------------------------------------------
Let's define the first convolution layer:

    layers {
      name: "conv1"
      type: CONVOLUTION
      blobs_lr: 1.
      blobs_lr: 2.
      convolution_param {
        num_output: 20
        kernelsize: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "data"
      top: "conv1"
    }

This layer takes the `data` blob (it is provided by the data layer), and produces the `conv1` layer. It produces outputs of 20 channels, with the convolutional kernel size 5 and carried out with stride 1.

The fillers allow us to randomly initialize the value of the weights and bias. For the weight filler, we will use the `xavier` algorithm that automatically determines the scale of initialization based on the number of input and output neurons. For the bias filler, we will simply initialize it as constant, with the default filling value 0.

`blobs_lr` are the learning rate adjustments for the layer's learnable parameters. In this case, we will set the weight learning rate to be the same as the learning rate given by the solver during runtime, and the bias learning rate to be twice as large as that - this usually leads to better convergence rates.

Writing the Pooling Layer
-------------------------
Phew. Pooling layers are actually much easier to define:

    layers {
      name: "pool1"
      type: POOLING
      pooling_param {
        kernel_size: 2
        stride: 2
        pool: MAX
      }
      bottom: "conv1"
      top: "pool1"
    }

This says we will perform max pooling with a pool kernel size 2 and a stride of 2 (so no overlapping between neighboring pooling regions).

Similarly, you can write up the second convolution and pooling layers. Check `data/lenet.prototxt` for details.

Writing the Fully Connected Layer
----------------------------------
Writing a fully connected layer is also simple:

    layers {
      name: "ip1"
      type: INNER_PRODUCT
      blobs_lr: 1.
      blobs_lr: 2.
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "pool2"
      top: "ip1"
    }

This defines a fully connected layer (for some legacy reason, Caffe calls it an `innerproduct` layer) with 500 outputs. All other lines look familiar, right?

Writing the ReLU Layer
----------------------
A ReLU Layer is also simple:

    layers {
      name: "relu1"
      type: RELU
      bottom: "ip1"
      top: "ip1"
    }

Since ReLU is an element-wise operation, we can do *in-place* operations to save some memory. This is achieved by simply giving the same name to the bottom and top blobs. Of course, do NOT use duplicated blob names for other layer types!

After the ReLU layer, we will write another innerproduct layer:

    layers {
      name: "ip2"
      type: INNER_PRODUCT
      blobs_lr: 1.
      blobs_lr: 2.
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "ip1"
      top: "ip2"
    }

Writing the Loss Layer
-------------------------
Finally, we will write the loss!

    layers {
      name: "loss"
      type: SOFTMAX_LOSS
      bottom: "ip2"
      bottom: "label"
    }

The `softmax_loss` layer implements both the softmax and the multinomial logistic loss (that saves time and improves numerical stability). It takes two blobs, the first one being the prediction and the second one being the `label` provided by the data layer (remember it?). It does not produce any outputs - all it does is to compute the loss function value, report it when backpropagation starts, and initiates the gradient with respect to `ip2`. This is where all magic starts.

Now that we have demonstrated how to write the MNIST layer definition prototxt, maybe check out [how we write a solver prototxt](mnist_solver_prototxt.html)?
