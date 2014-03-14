---
layout: default
title: Caffe
---

Welcome to Caffe
================

Caffe is a framework for convolutional neural network algorithms, developed with speed in mind.
It was created by [Yangqing Jia](http://www.eecs.berkeley.edu/~jiayq/) as a replacement of [decaf](http://decaf.berkeleyvision.org/), Yangqing's earlier Python implementation of CNNs.
It is maintained by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu) and several Berkeley vision group members are actively contributing to the codebase.

Caffe is released under [the BSD 2-Clause license](license.html).

Decaf, the big brother of Caffe, has a cool [demo](http://decaf.berkeleyvision.org). Caffe's own demo will come soon.

Why Caffe?
----------

Caffe aims to provide computer vision scientists with a **clean, modifiable implementation** of state-of-the-art deep learning algorithms.
For example, network structure is easily specified in separate config files, with no mess of hard-coded parameters in the code.

At the same time, Caffe fits industry needs, with blazing fast C++/Cuda code for GPU computation.
Caffe is currently the fastest GPU CNN implementation publicly available, and is able to process more than **20 million images per day** on a single Tesla K20 machine \*.

Caffe also provides **seamless switching between CPU and GPU**, which allows one to train models with fast GPUs and then deploy them on non-GPU clusters with one line of code: `Caffe::set_mode(Caffe::CPU)`.

Even in CPU mode, computing predictions on an image takes only 20 ms when images are processed in batch mode.

Quick Links
-----------

* [Presentation](caffe-presentation.pdf): The Caffe presentation, *updated 03/14*.
* [Installation](installation.html): Instructions on installing Caffe (tested on Ubuntu 12.04, but works on Red Hat, OS X, etc.).
* [Development](development.html): Guidelines for development and contributing to Caffe.
* [MNIST Demo](mnist.html): example of end-to-end training and testing on the MNIST data.
* [Training ImageNet](imagenet_training.html): tutorial on end-to-end training of an ImageNet classifier.
* [Running Pretrained ImageNet](imagenet_pretrained.html): simply runs in Python!
* [Running Detection](imagenet_detection.html): run a pretrained model as a detector.


Citing Caffe
------------
Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
       Year  = {2013},
       Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }

### Acknowledgements

Yangqing would like to thank the NVidia Academic program for providing a K20 GPU.
The Caffe Matlab wrapper is courtesy of [Dr. Ross Girshick](http://www.cs.berkeley.edu/~rbg/).
The detection module (`power_wrapper`) is courtesy of [Sergey Karayev](http://sergeykarayev.com/).
Our thanks also go to [Jeff Donahue](http://jeffdonahue.com/) and [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for various discussions along the journey.

---

\*: When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012.
More benchmarks coming soon.
