---
layout: default
title: Caffe
---

Welcome to Caffe
================

Caffe is a reference implementation for the recent convolutional neural networks algorithms with speed in mind. It is written and maintained by [Yangqing Jia](http://www.eecs.berkeley.edu/~jiayq/) as a replacement of [decaf](http://decaf.berkeleyvision.org/), the python implementation of convolutional neural networks. Several [Berkeley vision group](http://www.berkeelyvision.org/) members are actively contributing to the codebase.

Why Caffe?
----------

Caffe aims to expand deep learning research by providing computer vision researchers easier access to state-of-the-art deep learning implementations. At the same time, caffe also aims for fast computation speed that fits industry needs, with codes in C++/Cuda providing maximum performance through efficient GPU computations. Being able to process more than 20 million images per day\*, Caffe is currently **the fastest GPU CNN implementation publicly available**.

Caffe also provides **seamless switch between CPU and GPU implementations**, which allows one to train models with fast GPUs, but to still have the flexibility of deploying models on cheaper, non-GPU clusters, with only one line of code necessary:

```
Caffe::set_mode(Caffe::CPU);
```

Quick Links
-----------

* [Presentation](https://docs.google.com/presentation/d/1lzyXMRQFlOYE2Jy0lCNaqltpcCIKuRzKJxQ7vCuPRc8/edit?usp=sharing): Yangqing's presentation on Caffe at the Berkeley vision group meeting.
* [Installation](installation.html): expect some bumps along the ride.
* [Tutorial: MNIST](mnist.html): end-to-end training and testing on the MNIST data.
* [Reference ImageNet Model](imagenet.html): on how to run the ImageNet reference model.

Citing Caffe
------------
Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture
                for Feature Extraction},
       Year  = {2013},
       Howpublished = {\url{http://yangqing.github.io/caffe/}}

\* When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012, and run on a single Tesla K20 machine. Benchmark details coming soon.

\*\* We would like to thank the NVidia Academic gift program for donating GPUs for this project.