---
layout: default
---

# Pre-trained models

[BVLC](http://bvlc.eecs.berkeley.edu) aims to provide a variety of high quality pre-trained models.
Note that unlike Caffe itself, these models are licensed for **academic research / non-commercial use only**.
If you have any questions, please get in touch with us.

This page will be updated as more models become available.

### ImageNet

Our reference implementation of the AlexNet model trained on ILSVRC-2012 can be downloaded (232.57MB) by running `examples/imagenet/get_caffe_reference_imagenet_model.sh` from the Caffe root directory.

Additionally, you will probably eventually need some auxiliary data (mean image, synset list, etc.): run `data/ilsvrc12/get_ilsvrc_aux.sh` from the root directory to obtain it.
