---
layout: default
---

# Pre-trained models

[BVLC](http://bvlc.eecs.berkeley.edu) aims to provide a variety of high quality pre-trained models.
Note that unlike Caffe itself, these models are licensed for **academic research / non-commercial use only**.
If you have any questions, please get in touch with us.

This page will be updated as more models become available.

### ImageNet

**Caffe Reference ImageNet Model**: Our reference implementation of an ImageNet model trained on ILSVRC-2012 can be downloaded (232.6MB) by running `examples/imagenet/get_caffe_reference_imagenet_model.sh` from the Caffe root directory.

**AlexNet**: Our training of the Krizhevsky architecture, which differs from the paper's methodology by (1) not training with the relighting data-augmentation and (2) initializing non-zero biases to 0.1 instead of 1. (2) was found necessary for training, as initialization to 1 gave flat loss. Download the model (243.9MB) by running `examples/imagenet/get_caffe_alexnet_model.sh` from the Caffe root directory.

Additionally, you will probably eventually need some auxiliary data (mean image, synset list, etc.): run `data/ilsvrc12/get_ilsvrc_aux.sh` from the root directory to obtain it.
