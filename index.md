---
layout: default
title: Caffe
---

Welcome to Caffe
================

Caffe is a framework for the recent convolutional neural networks algorithms, developed with speed in mind. It is written and maintained by [Yangqing Jia](http://www.eecs.berkeley.edu/~jiayq/) as a replacement of [decaf](http://decaf.berkeleyvision.org/), the python implementation of CNNs. Several [Berkeley vision group](http://ucbvlc.org/) members are actively contributing to the codebase.

Caffe is currently released under [the UC Berkeley non-commercial license](license.html).

Why Caffe?
----------

Caffe aims to expand deep learning research by providing computer vision scientists easier access to state-of-the-art deep learning implementations. At the same time, caffe also aims for fast computation that fits industry needs, with codes in C++/Cuda providing maximum performance through efficient GPU computations. Being able to process more than **20 million images per day**\*, Caffe is currently the fastest GPU CNN implementation publicly available.

Caffe also provides **seamless switch between CPU and GPU**, which allows one to train models with fast GPUs, but to still have the flexibility of deploying models on cheaper, non-GPU clusters, with only one line of code necessary:

```
Caffe::set_mode(Caffe::CPU);
```

Quick Links
-----------

* [Presentation](https://docs.google.com/presentation/d/1lzyXMRQFlOYE2Jy0lCNaqltpcCIKuRzKJxQ7vCuPRc8/edit?usp=sharing): Yangqing's presentation on Caffe at the Berkeley vision group meeting.
* [Installation](installation.html): Instructions on installing Caffe, mainly with Ubuntu 12.04LTS.
* [MNIST Demo](mnist.html): end-to-end training and testing on the MNIST data.
* [Training ImageNet](imagenet.html): on how to train an ImageNet classifier.
* [Pretrained ImageNet](imagenet_pretrained.html): start running ImageNet classification in minutes.

Citing Caffe
------------
Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture
                for Feature Embedding},
       Year  = {2013},
       Howpublished = {\url{http://yangqing.github.io/caffe/}}

\* When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012, and run on a single machine with Intel i5 processor and Tesla K20. Benchmark details coming soon.

\*\* Yangqing would like to thank the NVidia Academic program for providing a K20 GPU.

\*\*\* Matlab wrapper courtsy of [Dr Ross Girshick](http://www.cs.berkeley.edu/~rbg/).
