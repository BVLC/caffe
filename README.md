[Caffe: Convolutional Architecture for Fast Feature Extraction](http://caffe.berkeleyvision.org)

Created by Yangqing Jia, Department of EECS, University of California, Berkeley.
Maintained by the Berkeley Vision and Learning Center (BVLC).

## Introduction

Caffe aims to provide computer vision scientists with a **clean, modifiable
implementation** of state-of-the-art deep learning algorithms. Network structure
is easily specified in separate config files, with no mess of hard-coded
parameters in the code. Python and Matlab wrappers are provided.

At the same time, Caffe fits industry needs, with blazing fast C++/Cuda code for
GPU computation. Caffe is currently the fastest GPU CNN implementation publicly
available, and is able to process more than **20 million images per day** on a
single Tesla K20 machine \*.

Caffe also provides **seamless switching between CPU and GPU**, which allows one
to train models with fast GPUs and then deploy them on non-GPU clusters with one
line of code: `Caffe::set_mode(Caffe::CPU)`.

Even in CPU mode, computing predictions on an image takes only 20 ms when images
are processed in batch mode.

* [Installation instructions](http://caffe.berkeleyvision.org/installation.html)
* [Caffe presentation](https://docs.google.com/presentation/d/1lzyXMRQFlOYE2Jy0lCNaqltpcCIKuRzKJxQ7vCuPRc8/edit?usp=sharing) at the Berkeley Vision Group meeting

\* When measured with the [SuperVision](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) model that won the ImageNet Large Scale Visual Recognition Challenge 2012.

## License

Caffe is BSD 2-Clause licensed (refer to the
[LICENSE](http://caffe.berkeleyvision.org/license.html) for details).

The pretrained models published by the BVLC, such as the
[Caffe reference ImageNet model](https://www.dropbox.com/s/n3jups0gr7uj0dv/caffe_reference_imagenet_model)
are licensed for academic research / non-commercial use only. However, Caffe is
a full toolkit for model training, so start brewing your own Caffe model today!

## Citing Caffe

Please kindly cite Caffe in your publications if it helps your research:

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}
    }
