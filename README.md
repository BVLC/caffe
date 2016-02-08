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
This fork is dedicated to improving Caffe performance when running on CPU, in particular Xeon
servers.

## Performance Results
Time measures are: average Forward-Backward as stated by *caffe time*.*speedup* is
(bvlc-caffe-master branch measure) / (intelcaffe-master branch measure)

### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (36 threads, MKL 11.3, GCC 4.8.3)
|            Branch | googlenet [ms] | caffenet [ms] | alexnet [ms] | cifar10-bn [ms] |
|------------------:|---------------:|--------------:|-------------:|----------------:|
| intelcaffe-master |            661 |          1237 |         1356 |               23|
| bvlc-caffe-master |           3872 |          6899 |         7343 |              323|
|    speedup factor |           x5.9 |          x5.6 |         x5.4 |            x14.0|

### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (36 thread, OpenBLAS 0.2.14, GCC 4.8.3)
|            Branch | googlenet [ms] | caffenet [ms] | alexnet [ms] | cifar10-bn [ms] |
|------------------:|---------------:|--------------:|-------------:|----------------:|
| intelcaffe-master |           1047 |          3004 |         3786 |               47|
| bvlc-caffe-master |          14892 |         25920 |        67542 |              530|
|    speedup factor |          x14.2 |          x8.6 |        x17.8 |            x11.3|

Tests were made using MKL and OpenBLAS. Please note that MKL is now available free of charge.
The speedup factor highly depends on the amount of running threads and system load.
Upper tables also shows, the optimal configuration is to use one thread per CPU core.

## Building
Build procedure is the same as on bvlc-caffe-master branch. Both Make and CMake can be used.
When OpenMP is available will be used automatically.

## Running
To utilize all hardware resources right, it is recommended to set OMP_NUM_THREADS environmental
variable to the number of all available CPU cores, which on Linux can be determined by typing
*lscpu* and is equal to number of "CPU socket(s)" times number of "Core(s) per socket".

It is also recommended to avoid movement of threads between processors by setting CPU affinity
which can be done by GOMP_CPU_AFFINITY environmental variable. It is achieved by setting its value
to "0-N", where N is total the number of physical cores MINUS ONE. For example a single
Intel(R) Xeon(R) E5-2699 v3 @ 2.30GHz have 18 CPU cores. When two sockets are used, there
will be in total 72 processing units. To bind threads to all 36 cores, a following command line
should look like:

    GOMP_CPU_AFFINITY="0-35" OMP_PROC_BIND="False" OMP_NUM_THREADS="36" ./build/tools/caffe time -iterations 50 --model=models/bvlc_googlenet/train_val.prototxt*  

If you have any questions, please do not hesitate to ask.

## Multinode Training

Please see the example how to run in examples/cifar10/train_full_multinode.sh.
The script will run data server, synchronous parameter server and 4 clients.
Prepared proto solvers should result in exactly the same behavior as single
node full cifar training.
The basic setup is to run parameter server with command like this:
"$TOOLS/caffe param_server --solver=/path/to/proto --listen_address=tcp://*:port"
Than run clients on machines you want with:
"$TOOLS/caffe train --solver=/path/to/proto --param_server:tcp://127.0.0.1:7777"

Data server is for convenience. By the default you could use data shard prepared
on each node separetely, either by shuffling the data uniquely or by creating
a subset of your training data. The remote data layer can be used to get data
from data server. It can also be used to cache data from the server in order
to reduce the network traffic.
In the case of choosing caching policy USE_CACHE_WHEN_FULL, it will first
download cache_size batches and then will randomized the cached data for actual
training.

The proto files need to be set up manually at the time, although you can use
model server to distribute some proto files among clients. To run model
server use the caffe tool similar to data server:
"$TOOLS/caffe model_server --solver=/path/to/proto --listen_address=tcp://*:6666"
To use the model in clients, replace the path to solver with model server
address: "$TOOLS/caffe train --solver=address"

Please see also prepared examples (for 2 nodes only) for googlenet in:
models/bvlc_googlenet/solver_param_server.prototxt 
models/bvlc_googlenet/solver_client.prototxt 
The solver tries to offset the bigger batch size with bigger learning rate.
According to paper 
    @article{
      Author = {Forrest N. Iandola, Khalid Ashraf, Matthew W. Moskewicz, Kurt Keutzer},
      Journal = {arXiv preprint arXiv:1511.00175},
      Title = {FireCaffe: near-linear acceleration of deep neural network training on compute clusters},
      Year = {2016}
    }
this should use 72 epochs to train googlenet. 

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
