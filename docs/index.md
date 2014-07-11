---
layout: default
---
# Caffe

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.
It was created by [Yangqing Jia](http://daggerfs.com), and is in active development by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and by community contributors.
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

## Why

**Clean architecture** enables rapid deployment.
Networks are specified in simple config files, with no hard-coded parameters in the code.
Switching between CPU and GPU code is as simple as setting a flag -- so models can be trained on a GPU machine, and then used on commodity clusters.

**Readable & modifiable implementation** fosters active development.
In Caffe's first six months, it has been forked by over 300 developers on Github, and many have contributed significant changes.

**Speed** makes Caffe perfect for industry use.
Caffe can process over **40M images per day** with a single NVIDIA K40 or Titan GPU\*.
That's 5 ms/image in training, and 2 ms/image in test.
We believe that Caffe is the fastest CNN implementation available.

**Community**: Caffe already powers academic research projects, startup prototypes, and even large-scale industrial applications in vision, speech, and multimedia.
There is an active discussion and support community on [Github](https://github.com/BVLC/caffe/issues).

<p class="footnote" markdown="1">
\* When files are properly cached, and using the ILSVRC2012-winning [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model.
Consult performance [details](/performance_hardware.html).
</p>

## How

* [Introductory slides](http://dl.caffe.berkeleyvision.org/caffe-presentation.pdf): slides about the Caffe architecture, *updated 03/14*.
* [ACM MM paper](http://ucb-icsi-vision-group.github.io/caffe-paper/caffe.pdf): a 4-page report for the ACM Multimedia Open Source competition.
* [Installation instructions](/installation.html): tested on Ubuntu, Red Hat, OS X.
* [Pre-trained models](/getting_pretrained_models.html): BVLC provides ready-to-use models for non-commercial use.
* [Development](/development.html): Guidelines for development and contributing to Caffe.

### Tutorials and Examples

* [Image Classification \[notebook\]][imagenet_classification]: classify images with the pretrained ImageNet model by the Python interface.
* [Detection \[notebook\]][detection]: run a pretrained model as a detector in Python.
* [Visualizing Features and Filters \[notebook\]][visualizing_filters]: extracting features and visualizing trained filters with an example image, viewed layer-by-layer.
* [Editing Model Parameters \[notebook\]][net_surgery]: how to do net surgery and manually change model parameters.
* [LeNet / MNIST Demo](/mnist.html): end-to-end training and testing of LeNet on MNIST.
* [CIFAR-10 Demo](/cifar10.html): training and testing on the CIFAR-10 data.
* [Training ImageNet](/imagenet_training.html): recipe for end-to-end training of an ImageNet classifier.
* [Feature extraction with C++](/feature_extraction.html): feature extraction using pre-trained model.

[imagenet_classification]:  http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/imagenet_classification.ipynb
[detection]:   http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb
[visualizing_filters]:  http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb
[net_surgery]:  http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb

## Citing Caffe

Please cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
       Author = {Yangqing Jia},
       Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
       Year  = {2013},
       Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }

If you do publish a paper where Caffe helped your research, we encourage you to update the [publications wiki](https://github.com/BVLC/caffe/wiki/Publications).
Citations are also tracked automatically by [Google Scholar](http://scholar.google.com/scholar?oi=bibs&hl=en&cites=17333247995453974016).

## Acknowledgements

Yangqing would like to thank the NVIDIA Academic program for providing GPUs, [Oriol Vinyals](http://www1.icsi.berkeley.edu/~vinyals/) for discussions along the journey, and BVLC PI [Trevor Darrell](http://www.eecs.berkeley.edu/~trevor/) for guidance.

A core set of BVLC members have contributed much new functionality and many fixes since the original release (alphabetical by first name):
[Eric Tzeng](https://github.com/erictzeng), [Evan Shelhamer](http://imaginarynumber.net/), [Jeff Donahue](http://jeffdonahue.com/), [Jon Long](https://github.com/longjon), [Ross Girshick](http://www.cs.berkeley.edu/~rbg/), [Sergey Karayev](http://sergeykarayev.com/), [Sergio Guadarrama](http://www.eecs.berkeley.edu/~sguada/).

Additionally, the open-source community plays a large and growing role in Caffe's development.
Check out the Github [project pulse](https://github.com/BVLC/caffe/pulse) for recent activity, and the [contributors](https://github.com/BVLC/caffe/graphs/contributors) for a sorted list.

We sincerely appreciate your interest and contributions!
If you'd like to contribute, please read the [development guide](development.html).
