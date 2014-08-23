---
title: Fine-tuning CaffeNet on "Flickr Style" data
description: We fine-tune the ImageNet-trained CaffeNet on another dataset.
category: example
include_in_docs: true
layout: default
priority: 5
---

# Fine-tuning CaffeNet on "Flickr Style" data

This example shows how to fine-tune the BVLC-distributed CaffeNet model on a different dataset: [Flickr Style](http://sergeykarayev.com/files/1311.3715v3.pdf), which has style category labels.

## Explanation

The Flickr-sourced data of the Style dataset is visually very similar to the ImageNet dataset, on which the `caffe_reference_imagenet_model` was trained.
Since that model works well for object category classification, we'd like to use it architecture for our style classifier.
We also only have 80,000 images to train on, so we'd like to start with the parameters learned on the 1,000,000 ImageNet images, and fine-tune as needed.
If we give provide the `model` parameter to the `caffe train` command, the trained weights will be loaded into our model, matching layers by name.

Because we are predicting 20 classes instead of a 1,000, we do need to change the last layer in the model.
Therefore, we change the name of the last layer from `fc8` to `fc8_flickr` in our prototxt.
Since there is no layer named that in the `caffe_reference_imagenet_model`, that layer will begin training with random weights.

We will also decrease the overall learning rate `base_lr` in the solver prototxt, but boost the `blobs_lr` on the newly introduced layer.
The idea is to have the rest of the model change very slowly with new data, but the new layer needs to learn fast.
Additionally, we set `stepsize` in the solver to a lower value than if we were training from scratch, since we're virtually far along in training and therefore want the learning rate to go down faster.
Note that we could also entirely prevent fine-tuning of all layers other than `fc8_flickr` by setting their `blobs_lr` to 0.

## Procedure

All steps are to be done from the root caffe directory.

The dataset is distributed as a list of URLs with corresponding labels.
Using a script, we will download a small subset of the data and split it into train and val sets.

    caffe % python examples/finetuning_on_flickr_style/assemble_data.py -h
    usage: assemble_data.py [-h] [-s SEED] [-i IMAGES] [-w WORKERS]

    Download a subset of Flickr Style to a directory

    optional arguments:
      -h, --help            show this help message and exit
      -s SEED, --seed SEED  random seed
      -i IMAGES, --images IMAGES
                            number of images to use (-1 for all)
      -w WORKERS, --workers WORKERS
                            num workers used to download images. -x uses (all - x)
                            cores.

    caffe % python examples/finetuning_on_flickr_style/assemble_data.py --workers=-1 --images=200
    Downloading 200 images with 7 workers...
    Writing train/val for 190 successfully downloaded images.

This script downloads images and writes train/val file lists into `data/flickr_style`.
The prototxt's in this example assume this, and also assume the presence of the ImageNet mean file (run `get_ilsvrc_aux.sh` from `data/ilsvrc12` to obtain this if you haven't yet).

We'll also need the ImageNet-trained model, which you can obtain by running `get_caffe_reference_imagenet_model.sh` from `examples/imagenet`.

Now we can train!

    caffe % ./build/tools/caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights examples/imagenet/caffe_reference_imagenet_model
    I0827 19:41:52.455621 2129298192 caffe.cpp:90] Starting Optimization
    I0827 19:41:52.456883 2129298192 solver.cpp:32] Initializing solver from parameters:

    [...]

    I0827 19:41:55.520205 2129298192 solver.cpp:46] Solver scaffolding done.
    I0827 19:41:55.520211 2129298192 caffe.cpp:99] Use CPU.
    I0827 19:41:55.520217 2129298192 caffe.cpp:107] Finetuning from examples/imagenet/caffe_reference_imagenet_model
    I0827 19:41:57.433044 2129298192 solver.cpp:165] Solving CaffeNet
    I0827 19:41:57.433104 2129298192 solver.cpp:251] Iteration 0, Testing net (#0)
    I0827 19:44:44.145447 2129298192 solver.cpp:302]     Test net output #0: accuracy = 0.004
    I0827 19:44:48.774271 2129298192 solver.cpp:195] Iteration 0, loss = 3.46922
    I0827 19:44:48.774333 2129298192 solver.cpp:397] Iteration 0, lr = 0.001
    I0827 19:46:15.107447 2129298192 solver.cpp:195] Iteration 20, loss = 0.0147678
    I0827 19:46:15.107511 2129298192 solver.cpp:397] Iteration 20, lr = 0.001
    I0827 19:47:41.941119 2129298192 solver.cpp:195] Iteration 40, loss = 0.00455839
    I0827 19:47:41.941181 2129298192 solver.cpp:397] Iteration 40, lr = 0.001

Note how rapidly the loss went down.
For comparison, here is how the loss goes down when we do not start with a pre-trained model:

    I0827 18:57:08.496208 2129298192 solver.cpp:46] Solver scaffolding done.
    I0827 18:57:08.496227 2129298192 caffe.cpp:99] Use CPU.
    I0827 18:57:08.496235 2129298192 solver.cpp:165] Solving CaffeNet
    I0827 18:57:08.496271 2129298192 solver.cpp:251] Iteration 0, Testing net (#0)
    I0827 19:00:00.894336 2129298192 solver.cpp:302]     Test net output #0: accuracy = 0.075
    I0827 19:00:05.825129 2129298192 solver.cpp:195] Iteration 0, loss = 3.51759
    I0827 19:00:05.825187 2129298192 solver.cpp:397] Iteration 0, lr = 0.001
    I0827 19:01:36.090224 2129298192 solver.cpp:195] Iteration 20, loss = 3.32227
    I0827 19:01:36.091948 2129298192 solver.cpp:397] Iteration 20, lr = 0.001
    I0827 19:03:08.522105 2129298192 solver.cpp:195] Iteration 40, loss = 2.97031
    I0827 19:03:08.522176 2129298192 solver.cpp:397] Iteration 40, lr = 0.001

## License

The Flickr Style dataset as distributed here contains only URLs to images.
Some of the images may have copyright.
Training a category-recognition model for research/non-commercial use may constitute fair use of this data.
