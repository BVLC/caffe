# Caffe
[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like
- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

# Intel Caffe
This fork is dedicated to improving Caffe performance when running on CPU, in particular Intel® Xeon processors (HSW, BDW, Xeon Phi)

## Building
Build procedure is the same as on bvlc-caffe-master branch. Both Make and CMake can be used.
When OpenMP is available will be used automatically.

## Running
Run procedure is the same as on bvlc-caffe-master branch.

Current implementation uses OpenMP threads. By default the number of OpenMP threads is set
to the number of CPU cores. Each one thread is bound to a single core to achieve best
performance results. It is however possible to use own configuration by providing right
one through OpenMP environmental variables like OMP_NUM_THREADS or GOMP_CPU_AFFINITY.

If some system tool like numactl is used to control CPU affinity, by default caffe will prevent
to use more than one thread per core. When less than required cores are specified, caffe will
limit execution of OpenMP threads to specified cores only.

## Best performance solution
Please read [release notes](https://github.com/intel/caffe/blob/master/docs/release_notes.md) for our recommendations and configuration to achieve best performance on Intel CPUs. 

## Multinode Training
Intel Caffe multinode allows you to execute deep neural network training on multiple machines.

To understand how it works and read some tutorials, go to our Wiki. Start from https://github.com/intelcaffe/caffe/wiki/Multinode-guide.

## License and Citation
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
