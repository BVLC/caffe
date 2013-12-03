---
layout: default
title: Caffe
---

Training MNIST with Caffe
================

We will assume that you have caffe successfully compiled. If not, please refer to the [Installation page](installation.html). In this tutorial, we will assume that your caffe installation is located at `CAFFE_ROOT`.

Prepare Datasets
----------------

You will first need to download and convert the data format from the MNIST website. To do this, simply run the following script:

    cd $CAFFE_ROOT/data
    ./get_mnist.sh

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively.

After running the scripts there should be two datasets, `CAFFE_ROOT/data/mnist-train-leveldb`, and `CAFFE_ROOT/data/mnist-test-leveldb`.

LeNet: the MNIST Classification Model
-------------------------------------
Before we actually run the training program, let's explain what will happen. We will use the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) network, which is known to work well on digit classification tasks. We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations with Rectified Linear Unit (ReLU) activations for the neurons.

This sections is under construction.
