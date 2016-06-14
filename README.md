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
This fork is dedicated to improving Caffe performance when running on CPU, in particular Intel® Xeon processors (HSW+)

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

## Multinode Training
Please see the example how to run in examples/cifar10/train_full_multinode.sh.
The script will run data server, synchronous parameter server and 4 clients.
Prepared proto solvers should result in exactly the same behavior as single
node full cifar training.
The basic setup is to run parameter server with command like this:
"$TOOLS/caffe param_server --solver=/path/to/proto --listen_address=tcp://*:port"
Than run clients on machines you want with:
"$TOOLS/caffe train --solver=/path/to/proto --param_server=tcp://127.0.0.1:7777"
The udp protocol can be used as well, for point to point communication
and with multicast (i.e. "udp://127.0.0.1:7777;239.1.1.1:7778").
It is also possible to run the scheme with mpi with mpirun command, i.e:
"mpirun \
    -host localhost -n 1 \
    $TOOLS/caffe param_server --solver=/path/to/proto --listen_address=mpi://uid \
  : \
    -host localhost -n 1 \
    $TOOLS/caffe train --solver=/path/to/proto --param_server=mpi://uid"
You can run relay points, to accumulate/broadcast data in a tree structure:
"$TOOLS/caffe param_server --solver=/path/to/proto --listen_address=tcp://*:port --param_server=tcp://127.0.0.1:7777"
It only works with tcp protocol.

The mpi setup with explicit all reduce is with command like this:
"mpirun -host 127.0.0.1 -n 5 $TOOLS/caffe train --solver=/path/to/proto --param_server=mpi"
This will run 5 processes on hosts set with host, and each process will calculate
it's own gradients, and propagate it up with a tree structure to the root, which
will apply them and propagate parameters down also in a tree structure.
This version is less configurable than one with param server, relay and client,
however it uses less cpu resource per node and can get most of the mpi implementations.

Data server is for convenience. By the default you could use data shard prepared
on each node separetely, either by shuffling the data uniquely or by creating
a subset of your training data. The remote data layer can be used to get data
from data server. It can also be used to cache data from the server in order
to reduce the network traffic. Use only tcp protocol with data server.
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
