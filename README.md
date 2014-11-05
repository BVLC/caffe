# Caffe with weighted samples

This is a branch of Caffe that allows training with weighted samples. The
branch is experimental, so not every layer is updated to fit this new paradigm, nor
are unit tests updated (yet). It might also easily break, since it has not be
throughly tested.

## How it works

The input data in Caffe is normally an associate array with the keys `"data"` and
`"label"`. The sample weights are input through this data structure by adding
another key called `"sample_weight"`. The weights should have the same shape as
the labels.

Now, `sample_weight` can be accessed just like `data` and `label`, so we need to make
sure our network's data layer loads them in:

    layers { # Your data layer
        # ...
        top: "data"
        top: "label"
        top: "sample_weight"
    }

Connect them to your loss layer:

    layers { # Your loss layer
        name: "loss"
        type: SOFTMAX_LOSS
        bottom: "ip1"  # or whatever name it might have
        bottom: "label"
        bottom: "sample_weight"
        top: "loss"
    }

## Features

Read `sample_weight` through:

* HDF5
* lmdb (untested)
* leveldb (untested)

The layers that have been made to appreciate `sample_weight` are:

* SOFTMAX_WITH_LOSS (`label` and `sample_weight` should be 1D)
* EUCLIDEAN_LOSS (`label` and `sample_weight` should be 2D)

For now, `sample_weight` is required to be specified, both in the training and
testing data (even though it is not used in the latter).

## Original Caffe note

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.<br />
Consult the [project website](http://caffe.berkeleyvision.org) for all documentation.
