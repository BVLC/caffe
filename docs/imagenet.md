---
layout: default
title: Caffe
---

Yangqing's Recipe on Brewing ImageNet
=====================================

    "All your braincells are belong to us."
        - Starbucks

We are going to describe a reference implementation for the approach first proposed by Krizhevsky, Sutskever, and Hinton in their [NIPS 2012 paper](http://books.nips.cc/papers/files/nips25/NIPS2012_0534.pdf). Since training the whole model takes quite some time and energy, we also provide a model, trained in the same way as we describe here, to help fight global warming. If you would like to simply use the pretrained model, check out the [Pretrained ImageNet](imagenet_pretrained.html) page.

To clarify, by ImageNet we actually mean the ILSVRC challenge, but you can easily train on the whole imagenet as well, just more disk space, and a little longer training time.

(If you don't get the quote, visit [Yann LeCun's fun page](http://yann.lecun.com/ex/fun/).

Data Preparation
----------------

We assume that you already have downloaded the ImageNet training data and validation data, and they are stored on your disk like:

    /path/to/imagenet/train/n01440764/n01440764_10026.JPEG
    /path/to/imagenet/val/ILSVRC2012_val_00000001.JPEG

You will first need to create a text file listing all the files as well as their labels. An example could be found in the caffe repo at `python/caffe/imagenet/ilsvrc_2012_train.txt` and `ilsvrc_2012_val.txt`. Note that in those two files we used a different indexing from the ILSVRC devkit: we sorted the synset names in their ASCII order, and then labeled them from 0 to 999.

You will also need to resize the images to 256x256: we do not explicitly do this because in a cluster environment, one may benefit from resizing images in a parallel fashion, using mapreduce. For example, Yangqing used his lightedweighted [mincepie](https://github.com/Yangqing/mincepie) package to do mapreduce on the Berkeley cluster. If you would things to be rather simple and straightforward, you can also use shell commands, something like:

    for name in /path/to/imagenet/val/*.JPEG; do
        convert -resize 256x256\! $name $name
    done

Now, you can simply create a leveldb using commands as follows:

    GLOG_logtostderr=1 examples/convert_imageset.bin \
        /path/to/imagenet/train/ \
        python/caffe/imagenet/ilsvrc_2012_train.txt \
        /path/to/imagenet-train-leveldb 1

Note that `/path/to/imagenet-train-leveldb` should not exist before this execution. It will be created by the script. `GLOG_logtostderr=1` simply dumps more information for you to inspect, and you can safely ignore it.

Compute Image Mean
------------------

The Model requires us to subtract the image mean from each image, so we have to compute the mean. `examples/demo_compute_image_mean.cpp` implements that - it is also a good example to familiarize yourself on how to manipulate the multiple components, such as protocol buffers, leveldbs, and logging, if you are not familiar with it. Anyway, the mean computation can be carried out as:

    examples/demo_compute_image_mean.bin /path/to/imagenet-train-leveldb /path/to/mean.binaryproto

where `/path/to/mean.binaryproto` will be created by the program.

Network Definition
------------------
The network definition follows strictly the one in Krizhevsky et al. You can find the detailed definition at `examples/imagenet.prototxt`. Note that to run it, you will most likely need to change the paths in the data layer - change the following lines

    source: "/home/jiayq/Data/ILSVRC12/train-leveldb"
    meanfile: "/home/jiayq/Data/ILSVRC12/image_mean.binaryproto"

to point to your own leveldb and image mean. Likewise, do the same for `examples/imagenet_val.prototxt`.

If you look carefully at `imagenet.prototxt` and `imagenet_val.prototxt`, you will notice that they are largely the same, with the only difference being the data layer sources, and the last layer: in training, we will be using a `softmax_loss` layer to compute the loss function and to initialize the backpropagation, while in validation we will be using an `accuracy` layer to inspect how well we do in terms of accuracy.

We will also lay out a protocol buffer for running the solver. Let's make a few plans:
* We will run in batches of 256, and run a total of 4,500,000 iterations (about 90 epochs).
* For every 1,000 iterations, we test the learned net on the validation data.
* We set the initial learning rate to 0.01, and decrease it every 100,000 iterations (about 20 epochs).
* Information will be displayed every 20 epochs.
* The network will be trained with momentum 0.9 and a weight decay of 0.0005.
* For every 10,000 iterations, we will take a snapshot of the current status.

Sounds good? This is implemented in `examples/imagenet_solver.prototxt`. Again, you will need to change the first two lines:

    train_net: "examples/imagenet.prototxt"
    test_net: "examples/imagenet_val.prototxt"

to point to the actual path.

Training ImageNet
-----------------

Ready? Let's train.

    GLOG_logtostderr=1 examples/train_net.bin examples/imagenet_solver.prototxt

Sit back and enjoy! On my K20 machine, every 20 iterations take about 36 seconds to run, so effectively about 7 ms per image for the full forward-backward pass. About 2.5 ms of this is on forward, and the rest is backward. If you are interested in dissecting the computation time, you can look at `examples/net_speed_benchmark.cpp`, but it was written purely for debugging purpose, so you may need to figure a few things out yourself.

Resume Training?
----------------

We all experience times when the power goes out, or we feel like rewarding ourself a little by playing Battlefield (does someone still remember Quake?). Since we are snapshotting intermediate results during training, we will be able to resume from snapshots. This can be done as easy as:

    GLOG_logtostderr=1 examples/train_net.bin examples/imagenet_solver.prototxt caffe_imagenet_train_10000.solverstate

where `caffe_imagenet_train_1000.solverstate` is the solver state snapshot that stores all necessary information to recover the exact solver state (including the parameters, momentum history, etc).

Parting Words
-------------

Hope you liked this recipe. Many researchers have gone further since the ILSVRC 2012 challenge, changing the network architecture and/or finetuning the various parameters in the network. The recent ILSVRC 2013 challenge suggests that there are quite some room for improvement. **Caffe allows one to explore different network choices  more easily, by simply writing different prototxt files** - isn't that exciting?

And since now you have a trained network, check out how to use it: [Running Pretrained ImageNet](imagenet_pretrained.html). This time we will use Python, but if you have wrappers for other languages, please kindly send me a pull request!
