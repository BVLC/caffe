---
layout: default
title: Caffe
---

Yangqing's Recipe on Brewing ImageNet
=====================================

    "All your braincells are belong to us."
        - Caffeine

We are going to describe a reference implementation for the approach first proposed by Krizhevsky, Sutskever, and Hinton in their [NIPS 2012 paper](http://books.nips.cc/papers/files/nips25/NIPS2012_0534.pdf). Since training the whole model takes some time and energy, we provide a model, trained in the same way as we describe here, to help fight global warming. If you would like to simply use the pretrained model, check out the [Pretrained ImageNet](getting_pretrained_models.html) page. *Note that the pretrained model is for academic research / non-commercial use only*.

To clarify, by ImageNet we actually mean the ILSVRC12 challenge, but you can easily train on the whole of ImageNet as well, just with more disk space, and a little longer training time.

(If you don't get the quote, visit [Yann LeCun's fun page](http://yann.lecun.com/ex/fun/).

Data Preparation
----------------

We assume that you already have downloaded the ImageNet training data and validation data, and they are stored on your disk like:

    /path/to/imagenet/train/n01440764/n01440764_10026.JPEG
    /path/to/imagenet/val/ILSVRC2012_val_00000001.JPEG

You will first need to prepare some auxiliary data for training. This data can be downloaded by:

    cd $CAFFE_ROOT/data/ilsvrc12/
    ./get_ilsvrc_aux.sh

The training and validation input are described in `train.txt` and `val.txt` as text listing all the files and their labels. Note that we use a different indexing for labels than the ILSVRC devkit: we sort the synset names in their ASCII order, and then label them from 0 to 999. See `synset_words.txt` for the synset/name mapping.

You will also need to resize the images to 256x256: we do not explicitly do this because in a cluster environment, one may benefit from resizing images in a parallel fashion, using mapreduce. For example, Yangqing used his lightedweighted [mincepie](https://github.com/Yangqing/mincepie) package to do mapreduce on the Berkeley cluster. If you would things to be rather simple and straightforward, you can also use shell commands, something like:

    for name in /path/to/imagenet/val/*.JPEG; do
        convert -resize 256x256\! $name $name
    done

Go to `$CAFFE_ROOT/examples/imagenet/` for the rest of this guide.

Take a look at `create_imagenet.sh`. Set the paths to the train and val dirs as needed. Now simply create the leveldbs with `./create_imagenet.sh`. Note that `imagenet_train_leveldb` and `imagenet_val_leveldb` should not exist before this execution. It will be created by the script. `GLOG_logtostderr=1` simply dumps more information for you to inspect, and you can safely ignore it.

Compute Image Mean
------------------

The model requires us to subtract the image mean from each image, so we have to compute the mean. `tools/compute_image_mean.cpp` implements that - it is also a good example to familiarize yourself on how to manipulate the multiple components, such as protocol buffers, leveldbs, and logging, if you are not familiar with them. Anyway, the mean computation can be carried out as:

    ./make_imagenet_mean.sh

which will make `data/ilsvrc12/imagenet_mean.binaryproto`.

Network Definition
------------------

The network definition follows strictly the one in Krizhevsky et al. You can find the detailed definition at `examples/imagenet/imagenet_train.prototxt`. Note the paths in the data layer - if you have not followed the exact paths in this guide you will need to change the following lines:

    source: "ilvsrc12_train_leveldb"
    mean_file: "../../data/ilsvrc12/imagenet_mean.binaryproto"

to point to your own leveldb and image mean. Likewise, do the same for `examples/imagenet/imagenet_val.prototxt`.

If you look carefully at `imagenet_train.prototxt` and `imagenet_val.prototxt`, you will notice that they are largely the same, with the only difference being the data layer sources, and the last layer: in training, we will be using a `softmax_loss` layer to compute the loss function and to initialize the backpropagation, while in validation we will be using an `accuracy` layer to inspect how well we do in terms of accuracy.

We will also lay out a protocol buffer for running the solver. Let's make a few plans:
* We will run in batches of 256, and run a total of 4,500,000 iterations (about 90 epochs).
* For every 1,000 iterations, we test the learned net on the validation data.
* We set the initial learning rate to 0.01, and decrease it every 100,000 iterations (about 20 epochs).
* Information will be displayed every 20 epochs.
* The network will be trained with momentum 0.9 and a weight decay of 0.0005.
* For every 10,000 iterations, we will take a snapshot of the current status.

Sound good? This is implemented in `examples/imagenet/imagenet_solver.prototxt`. Again, you will need to change the first two lines:

    train_net: "imagenet_train.prototxt"
    test_net: "imagenet_val.prototxt"

to point to the actual path if you have changed them.

Training ImageNet
-----------------

Ready? Let's train.

    ./train_imagenet.sh

Sit back and enjoy! On my K20 machine, every 20 iterations take about 36 seconds to run, so effectively about 7 ms per image for the full forward-backward pass. About 2.5 ms of this is on forward, and the rest is backward. If you are interested in dissecting the computation time, you can look at `examples/net_speed_benchmark.cpp`, but it was written purely for debugging purpose, so you may need to figure a few things out yourself.

Resume Training?
----------------

We all experience times when the power goes out, or we feel like rewarding ourself a little by playing Battlefield (does someone still remember Quake?). Since we are snapshotting intermediate results during training, we will be able to resume from snapshots. This can be done as easy as:

    ./resume_training.sh

where in the script `caffe_imagenet_train_1000.solverstate` is the solver state snapshot that stores all necessary information to recover the exact solver state (including the parameters, momentum history, etc).

Parting Words
-------------

Hope you liked this recipe! Many researchers have gone further since the ILSVRC 2012 challenge, changing the network architecture and/or finetuning the various parameters in the network. The recent ILSVRC 2013 challenge suggests that there are quite some room for improvement. **Caffe allows one to explore different network choices  more easily, by simply writing different prototxt files** - isn't that exciting?

And since now you have a trained network, check out how to use it: [Running Pretrained ImageNet](getting_pretrained_models.html). This time we will use Python, but if you have wrappers for other languages, please kindly send a pull request!
