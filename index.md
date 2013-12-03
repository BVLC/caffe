---
layout: default
title: Caffe
---

Welcome to Caffe
================

Caffe is a reference implementation for the recent convolutional neural networks algorithms with performance in mind. It is written in C++/Cuda to provide maximum performance through efficient GPU computations. It is written and maintained by [Yangqing Jia](http://www.eecs.berkeley.edu/~jiayq/) as a replacement of [decaf](http://decaf.berkeleyvision.org/), the python implementation of convolutional neural networks. Several [Berkeley vision group](http://www.berkeelyvision.org/) members are actively contributing to the codebase.

Why Caffe?
----------

As an open-source implementation of CNN, Caffe aims to enable more computer vision researchers to have access to the most recent convolutional neural networks algorithms (also known as deep learning). At the same time, caffe is designed to run fast, and is currently the fastest GPU implementation publicly available\*.

Caffe also provides a seamless switch between CPU and GPU implementations, which enables one to train models with fast GPU computation but to still have the flexibility of deploying the models on cheaper, non-GPU machines, with the only line of code required for the switch as simple as:

```
Caffe::set_mode(Caffe::CPU);
```



Quick Links
-----------

* [Installation](installation.html): expect some bumps along the ride.
* [Tutorial: MNIST](mnist.html): end-to-end training and testing on the MNIST data.
* [Reference ImageNet Model](imagenet.html): on how to run the ImageNet reference model.

\* When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012. Benchmark details coming soon.
