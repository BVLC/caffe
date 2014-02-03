# Installation

To build and test Caffe do

    cp Makefile.config.example Makefile.config
    make
    make test
    make runtest

You will probably need to adust paths in `Makefile.config` and maybe the
`Makefile` itself.
Feel free to issue a pull request for a change that may help other people.

Note that building and running CPU-only works, but GPU tests will naturally
fail.

We mostly used Ubuntu 12.04 for development, and here we describe the
step-to-step guide on installing Caffe on Ubuntu.

## Prerequisites

* CUDA
* Boost
* MKL (but see the boost-eigen branch for a boost/Eigen3 port)
* OpenCV
* glog, gflags, protobuf, leveldb, snappy
* For the Python wrapper: python, numpy (>= 1.7 preferred), and boost_python
* For the Matlab wrapper: Matlab with mex

Caffe requires the CUDA NVCC compiler to compile its GPU code. To install CUDA, go to the [NVidia CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. Caffe is verified to compile with both CUDA 5.0 and 5.5.

Caffe also needs Intel MKL as the backend of its matrix computation and vectorized computations. We are in the process of removing MKL dependency, but for now you will need to have an MKL installation. You can obtain a [trial license](http://software.intel.com/en-us/intel-mkl) or an [academic license](http://software.intel.com/en-us/intel-education-offerings) (if you are a student).

If you would like to compile the Python wrapper, you will need to install python, numpy and boost_python. You can either compile them from scratch or use a pre-packaged solution like [Anaconda](https://store.continuum.io/cshop/anaconda/) or [Enthought Canopy](https://www.enthought.com/products/canopy/). Note that if you use the Ubuntu default python, you will need to apt-install the `python-dev` package to have the python headers.

If you would like to compile the Matlab wrapper, you will need to install Matlab.

You will also need other packages, most of which can be installed via apt-get using:

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev

The only exception being the google logging library, which does not exist in the Ubuntu 12.04 repository. To install it, do:

    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    ./configure
    make && make install

After setting all the prerequisites, you should modify the `Makefile.config` file and change the paths to those on your computer.

## Compilation

After installing the prerequisites, simply do `make all` to compile Caffe. If you would like to compile the Python and Matlab wrappers, do

    make pycaffe
    make matcaffe

Optionally, you can run `make distribute` to create a `build` directory that contains all the necessary files, including the headers, compiled shared libraries, and binary files that you can distribute over different machines.

To use Caffe with python, you will need to add `/path/to/caffe/python` or `/path/to/caffe/build/python` to your `PYTHONPATH`.

Now that you have compiled Caffe, check out the [MNIST demo](mnist.html) and the pretrained [ImageNet example](imagenet.html).
