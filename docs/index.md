---
layout: default
---
# Welcome to Caffe

Caffe is a framework for convolutional neural network algorithms, developed with speed in mind.
It was created by [Yangqing Jia](http://daggerfs.com), and is in active development by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu).

Caffe is released under [the BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

<!-- BVLC hosts a quick [classification demo](http://demo.caffe.berkeleyvision.org/) using Caffe. -->

## Why Caffe?

Caffe aims to provide computer vision scientists and practitioners with a **clean and modifiable implementation** of state-of-the-art deep learning algorithms.
For example, network structure is easily specified in separate config files, with no mess of hard-coded parameters in the code.

At the same time, Caffe fits industry needs, with blazing fast C++/CUDA code for GPU computation.
Caffe is currently the fastest GPU CNN implementation publicly available, and is able to process more than **20 million images per day** on a single Tesla K20 machine \*.

Caffe also provides **seamless switching between CPU and GPU**, which allows one to train models with fast GPUs and then deploy them on non-GPU clusters with one line of code: `Caffe::set_mode(Caffe::CPU)`.
Even in CPU mode, computing predictions on an image takes only 20 ms when images are processed in batch mode.

## Documentation

* [Introductory slides](https://www.dropbox.com/s/10fx16yp5etb8dv/caffe-presentation.pdf): slides about the Caffe architecture, *updated 03/14*.
* [Installation](/installation.html): Instructions on installing Caffe (works on Ubuntu, Red Hat, OS X).
* [Pre-trained models](/getting_pretrained_models.html): BVLC provides some pre-trained models for academic / non-commercial use.
* [Development](/development.html): Guidelines for development and contributing to Caffe.

### Examples

* [LeNet / MNIST Demo](/mnist.html): end-to-end training and testing of LeNet on MNIST.
* [CIFAR-10 Demo](/cifar10.html): training and testing on the CIFAR-10 data.
* [Training ImageNet](/imagenet_training.html): end-to-end training of an ImageNet classifier.
* [Feature extraction with C++](/feature_extraction.html): feature extraction using pre-trained model
* [Running Pretrained ImageNet \[notebook\]][pretrained_imagenet]: run classification with the pretrained ImageNet model using the Python interface.
* [Running Detection \[notebook\]][imagenet_detection]: run a pretrained model as a detector.
* [Visualizing Features and Filters \[notebook\]][visualizing_filters]: trained filters and an example image, viewed layer-by-layer.

[pretrained_imagenet]:  http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/imagenet_pretrained.ipynb
[imagenet_detection]:   http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/selective_search_demo.ipynb
[visualizing_filters]:  http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

## Citing Caffe

Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
       Year  = {2013},
       Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }

### Acknowledgements

Yangqing would like to thank the NVidia Academic program for providing K20 GPUs, and [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for various discussions along the journey.

A core set of BVLC members have contributed lots of new functionality and fixes since the original release (alphabetical by first name):

- [Eric Tzeng](https://github.com/erictzeng)
- [Evan Shelhamer](http://imaginarynumber.net/)
- [Jeff Donahue](http://jeffdonahue.com/)
- [Jon Long](https://github.com/longjon)
- [Dr. Ross Girshick](http://www.cs.berkeley.edu/~rbg/)
- [Sergey Karayev](http://sergeykarayev.com/)
- [Dr. Sergio Guadarrama](http://www.eecs.berkeley.edu/~sguada/)

Additionally, the open-source community plays a large and growing role in Caffe's development.
Check out the Github [project pulse](https://github.com/BVLC/caffe/pulse) for recent activity, and the [contributors](https://github.com/BVLC/caffe/graphs/contributors) for an ordered list (by commit activity).
We sincerely appreciate your interest and contributions!
If you'd like to contribute, read [this](development.html).

---

\*: When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012.
