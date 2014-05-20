---
layout: default
title: Caffe
---

Training MNIST with Caffe
================

We will assume that you have caffe successfully compiled. If not, please refer to the [Installation page](installation.html). In this tutorial, we will assume that your caffe installation is located at `CAFFE_ROOT`.

Prepare Datasets
----------------

You will first need to download and convert the data format from the MNIST website. To do this, simply run the following commands:

    cd $CAFFE_ROOT/data/mnist
    ./get_mnist.sh
    cd $CAFFE_ROOT/examples/mnist
    ./create_mnist.sh

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be two datasets, `mnist-train-leveldb`, and `mnist-test-leveldb`.

LeNet: the MNIST Classification Model
-------------------------------------
Before we actually run the training program, let's explain what will happen. We will use the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) network, which is known to work well on digit classification tasks. We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations with Rectified Linear Unit (ReLU) activations for the neurons.

The design of LeNet contains the essence of CNNs that are still used in larger models such as the ones in ImageNet. In general, it consists of a convolutional layer followed by a pooling layer, another convolution layer followed by a pooling layer, and then two fully connected layers similar to the conventional multilayer perceptrons. We have defined the layers in `CAFFE_ROOT/data/lenet.prototxt`.

If you would like to read about step-by-step instruction on how the protobuf definitions are written, see [MNIST: Define the Network](mnist_prototxt.html) and [MNIST: Define the Solver](mnist_solver_prototxt.html)?.

Training and Testing the Model
------------------------------

Training the model is simple after you have written the network definition protobuf and solver protobuf files. Simply run `train_mnist.sh`, or the following command directly:

    cd $CAFFE_ROOT/examples/mnist
    ./train_lenet.sh

`train_lenet.sh` is a simple script, but here are a few explanations: `GLOG_logtostderr=1` is the google logging flag that prints all the logging messages directly to stderr. The main tool for training is `train_net.bin`, with the solver protobuf text file as its argument.

When you run the code, you will see a lot of messages flying by like this:

    I1203 net.cpp:66] Creating Layer conv1
    I1203 net.cpp:76] conv1 <- data
    I1203 net.cpp:101] conv1 -> conv1
    I1203 net.cpp:116] Top shape: 20 24 24
    I1203 net.cpp:127] conv1 needs backward computation.

These messages tell you the details about each layer, its connections and its output shape, which may be helpful in debugging. After the initialization, the training will start:

    I1203 net.cpp:142] Network initialization done.
    I1203 solver.cpp:36] Solver scaffolding done.
    I1203 solver.cpp:44] Solving LeNet

Based on the solver setting, we will print the training loss function every 100 iterations, and test the network every 1000 iterations. You will see messages like this:

    I1203 solver.cpp:204] Iteration 100, lr = 0.00992565
    I1203 solver.cpp:66] Iteration 100, loss = 0.26044
    ...
    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9785
    I1203 solver.cpp:111] Test score #1: 0.0606671

For each training iteration, `lr` is the learning rate of that iteration, and `loss` is the training function. For the output of the testing phase, score 0 is the accuracy, and score 1 is the testing loss function.

And after a few minutes, you are done!

    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9897
    I1203 solver.cpp:111] Test score #1: 0.0324599
    I1203 solver.cpp:126] Snapshotting to lenet_iter_10000
    I1203 solver.cpp:133] Snapshotting solver state to lenet_iter_10000.solverstate
    I1203 solver.cpp:78] Optimization Done.

The final model, stored as a binary protobuf file, is stored at

    lenet_iter_10000

which you can deploy as a trained model in your application, if you are training on a real-world application dataset.

Um... How about GPU training?
-----------------------------

You just did! All the training was carried out on the GPU. In fact, if you would like to do training on CPU, you can simply change one line in `lenet_solver.prototxt`:

    # solver mode: CPU or GPU
    solver_mode: CPU

and you will be using CPU for training. Isn't that easy?

MNIST is a small dataset, so training with GPU does not really introduce too much benefit due to communication overheads. On larger datasets with more complex models, such as ImageNet, the computation speed difference will be more significant.
