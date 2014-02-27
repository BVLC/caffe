---
layout: default
title: Caffe
---

Define the MNIST Solver
=======================

The page is under construction. For now, check out the comments in the solver prototxt file, which explains each line in the prototxt:

    # The training protocol buffer definition
    train_net: "lenet_train.prototxt"
    # The testing protocol buffer definition
    test_net: "lenet_test.prototxt"
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    test_iter: 100
    # Carry out testing every 500 training iterations.
    test_interval: 500
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
    # The learning rate policy
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # Display every 100 iterations
    display: 100
    # The maximum number of iterations
    max_iter: 10000
    # snapshot intermediate results
    snapshot: 5000
    snapshot_prefix: "lenet"
    # solver mode: 0 for CPU and 1 for GPU
    solver_mode: 1
