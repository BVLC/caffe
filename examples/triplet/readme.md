---
title: Triplet Network Tutorial
description: Train and test a triplet network on MNIST data.
category: example
include_in_docs: true
layout: default
priority: 100
---

# Triplet Network Training with Caffe
This example shows how you can use weight sharing and a contrastive loss
function to learn a model using a triplet network in Caffe.

We will assume that you have caffe successfully compiled. If not, please refer
to the [Installation page](../../installation.html). This example builds on the
[MNIST tutorial](mnist.html) so it would be a good idea to read that before
continuing.

*The guide specifies all paths and assumes all commands are executed from the
root caffe directory*

## Prepare Datasets

You will first need to download and convert the data from the MNIST
website. To do this, simply run the following commands:

    ./data/mnist/get_mnist.sh
    ./examples/triplet/create_mnist_triplet.sh

After running the script there should be two datasets,
`./examples/triplet/mnist_triplet_train_leveldb`, and
`./examples/triplet/mnist_triplet_test_leveldb`.

## The Model
First, we will define the model that we want to train using the triplet network.
We will use the convolutional net defined in
`./examples/triplet/mnist_triplet.prototxt`. This model is almost
exactly the same as the [LeNet model](mnist.html), the only difference is that
we have replaced the top layers that produced probabilities over the 10 digit
classes with a linear "feature" layer that produces a 2 dimensional vector.

    layers {
      name: "feat"
      type: INNER_PRODUCT
      bottom: "ip2"
      top: "feat"
      blobs_lr: 1
      blobs_lr: 2
      inner_product_param {
        num_output: 2
      }
    }

## Define the triplet Network

In this section we will define the triplet network used for training. The
resulting network is defined in
`./examples/triplet/mnist_triplet_train_test.prototxt`.

### Reading in the Triplet Data

We start with a data layer that reads from the LevelDB database we created
earlier. Each entry in this database contains the image data for a triplet of
images (`triplet_data`) and the label (`sim`) is not nessesary in our method.

    layers {
      name: "triplet_data"
      type: DATA
      top: "triplet_data"
      top: "sim"
      data_param {
        source: "examples/triplet/mnist-triplet-train-leveldb"
        scale: 0.00390625
        batch_size: 64
      }
      include: { phase: TRAIN }
    }

In order to pack a triplet of images into the same blob in the database we pack one
image per channel. We want to be able to work with these three images separately,
so we add a slice layer after the data layer. This takes the `triplet_data` and
slices it along the channel dimension so that we have a single image in `data`
and its positive image in `data_pos.` & its negative image in `data_neg.`

    layers {
        name: "slice_triplet"
        type: SLICE
        bottom: "triplet_data"
        top: "data"
        top: "data_pos"
        top: "data_neg"
        slice_param {
        slice_dim: 1
        slice_point: 1
        slice_point: 2
  }
    }

### Building the First part of the triplet Net

Now we can specify the first side of the triplet net. This side operates on
`data` and produces `feat`. Starting from the net in
`./examples/triplet/mnist_triplet.prototxt` we add default weight fillers. Then
we name the parameters of the convolutional and inner product layers. Naming the
parameters allows Caffe to share the parameters between layers on three channels of
the triplet net. In the definition this looks like:

    ...
    param: "conv1_w"
    param: "conv1_b"
    ...
    param: "conv2_w"
    param: "conv2_b"
    ...
    param: "ip1_w"
    param: "ip1_b"
    ...
    param: "ip2_w"
    param: "ip2_b"
    ...

### Building the Second Side of the triplet Net

Now we need to create the second path that operates on `data_pos` and produces
`feat_pos`. This path is exactly the same as the first. So we can just copy and
paste it. Then we change the name of each layer, input, and output by appending
`_pos` to differentiate the "paired" layers from the originals.

### Building the Third Side of the triplet Net

Now we need to create the second path that operates on `data_neg` and produces
`feat_neg`. This path is exactly the same as the first. So we can just copy and
paste it. Then we change the name of each layer, input, and output by appending
`_neg` to differentiate the "paired" layers from the originals.

### Adding the Triplet Loss Function

To train the network we will optimize a triplet loss function proposed in:
This cost function is implemented with the `TRIPLET_LOSS` layer:

    layers {
        name: "loss"
        type: TRIPLET_LOSS
        triplet_loss_param {
            margin: 0.2
        }
        bottom: "feat"
        bottom: "feat_pos"
        bottom: "feat_neg"
        bottom: "sim"
        top: "loss"
    }

## Define the Solver

Nothing special needs to be done to the solver besides pointing it at the
correct model file. The solver is defined in
`./examples/triplet/mnist_triplet_solver.prototxt`.

## Training and Testing the Model

Training the model is simple after you have written the network definition
protobuf and solver protobuf files. Simply run
`./examples/triplet/train_mnist_triplet.sh`:

    ./examples/triplet/train_mnist_triplet.sh

# Plotting the results

First, we can draw the model and triplet networks by running the following
commands that draw the DAGs defined in the .prototxt files:

    ./python/draw_net.py \
        ./examples/triplet/mnist_triplet.prototxt \
        ./examples/triplet/mnist_triplet.png

    ./python/draw_net.py \
        ./examples/triplet/mnist_triplet_train_test.prototxt \
        ./examples/triplet/mnist_triplet_train_test.png

Second, we can load the learned model and plot the features using the iPython
notebook:

    ipython notebook ./examples/triplet/mnist_triplet.ipynb

