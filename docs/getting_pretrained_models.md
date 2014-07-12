---
layout: default
---

# Pre-trained models

[BVLC](http://bvlc.eecs.berkeley.edu) aims to provide a variety of high quality pre-trained models.
Note that unlike Caffe itself, these models are licensed for **academic research / non-commercial use only**.
If you have any questions, please get in touch with us.

*UPDATE* July 2014: we are actively working on a service for hosting user-uploaded model definition and trained weight files.
Soon, the community will be able to easily contribute different architectures!

### ImageNet

**Caffe Reference ImageNet Model**: Our reference implementation of an ImageNet model trained on ILSVRC-2012 can be downloaded (232.6MB) by running `examples/imagenet/get_caffe_reference_imagenet_model.sh` from the Caffe root directory.

- The bundled model is the iteration 310,000 snapshot.
- The best validation performance during training was iteration 313,000 with
  validation accuracy 57.412% and loss 1.82328.
- This model obtains a top-1 accuracy 57.4% and a top-5 accuracy 80.4% on the validation set, using just the center crop. (Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy)

**AlexNet**: Our training of the Krizhevsky architecture, which differs from the paper's methodology by (1) not training with the relighting data-augmentation and (2) initializing non-zero biases to 0.1 instead of 1. (2) was found necessary for training, as initialization to 1 gave flat loss. Download the model (243.9MB) by running `examples/imagenet/get_caffe_alexnet_model.sh` from the Caffe root directory.

- The bundled model is the iteration 360,000 snapshot.
- The best validation performance during training was iteration 358,000 with
  validation accuracy 57.258% and loss 1.83948.
- This model obtains a top-1 accuracy 57.1% and a top-5 accuracy 80.2% on the validation set, using just the center crop. (Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy)

### Auxiliary Data

Additionally, you will probably eventually need some auxiliary data (mean image, synset list, etc.): run `data/ilsvrc12/get_ilsvrc_aux.sh` from the root directory to obtain it.
