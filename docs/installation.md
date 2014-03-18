---
layout: default
title: Caffe
---

# Installation

Prior to installing, it is best to read through this guide and take note of the details for your platform. We mostly develop and deploy on Ubuntu 12.04, although we have also installed on OS X 10.8 (and 10.9 with further effort) through homebrew.

- [Prerequisites](#prequequisites)
- [Compilation](#compilation)
- [OS X installation](#os_x_installation)
- [Hardware questions](#hardware_questions)

To build and test Caffe do

    cp Makefile.config.example Makefile.config
    make
    make test
    make runtest

You will probably need to adust paths in `Makefile.config` and maybe the `Makefile` itself. Feel free to issue a pull request for a change that may help other people.

Note that building and running CPU-only works, but GPU tests will naturally fail.

The following sections detail prerequisites and installation on Ubuntu. For OS X notes, refer to the table of contents above to skip ahead.

## Prerequisites

* CUDA (5.0 or 5.5)
* Boost
* MKL (but see the [boost-eigen branch](https://github.com/BVLC/caffe/tree/boost-eigen) for a boost/Eigen3 port)
* OpenCV
* glog, gflags, protobuf, leveldb, snappy, hdf5
* For the Python wrapper: python, numpy (>= 1.7 preferred), and boost_python
* For the Matlab wrapper: Matlab with mex

Caffe requires the CUDA NVCC compiler to compile its GPU code. To install CUDA, go to the [NVidia CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. Caffe is verified to compile with both CUDA 5.0 and 5.5.

N.B. one can install the CUDA libraries without the CUDA driver in order to build and run Caffe in CPU-only mode.

Caffe also needs Intel MKL as the backend of its matrix computation and vectorized computations. We are in the process of removing MKL dependency, but for now you will need to have an MKL installation. You can obtain a [trial license](http://software.intel.com/en-us/intel-mkl) or an [academic license](http://software.intel.com/en-us/intel-education-offerings) (if you are a student).

You will also need other packages, most of which can be installed via apt-get using:

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

The only exception being the google logging library, which does not exist in the Ubuntu 12.04 repository. To install it, do:

    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    ./configure
    make && make install

If you would like to compile the Python wrapper, you will need to install python, numpy and boost_python. You can either compile them from scratch or use a pre-packaged solution like [Anaconda](https://store.continuum.io/cshop/anaconda/) or [Enthought Canopy](https://www.enthought.com/products/canopy/). Note that if you use the Ubuntu default python, you will need to apt-install the `python-dev` package to have the python headers. You can install any remaining dependencies with

    pip install -r /path/to/caffe/python/requirements.txt

If you would like to compile the Matlab wrapper, you will need to install Matlab.

After setting all the prerequisites, you should modify the `Makefile.config` file and change the paths to those on your computer.

## Compilation

After installing the prerequisites, simply do `make all` to compile Caffe. If you would like to compile the Python and Matlab wrappers, do

    make pycaffe
    make matcaffe

Optionally, you can run `make distribute` to create a `distribute` directory that contains all the necessary files, including the headers, compiled shared libraries, and binary files that you can distribute over different machines.

To use Caffe with python, you will need to add `/path/to/caffe/python` or `/path/to/caffe/build/python` to your `PYTHONPATH`.

Now that you have compiled Caffe, check out the [MNIST demo](mnist.html) and the pretrained [ImageNet example](imagenet.html).

## OS X Installation

On 10.8, we have successfully compiled and run Caffe on GPU-equipped Macbook Pros. Caffe also runs on 10.9, but you need to do a few extra steps described below.

### Install prerequisites using Homebrew

Install [homebrew](http://brew.sh/) to install most of the prerequisites. Starting from a clean install of the OS (or from a wiped `/usr/local`) is recommended to avoid conflicts. For python, [Anaconda](https://store.continuum.io/cshop/anaconda/) and homebrew python are confirmed to work.

    # install python by (1) Anaconda or (2) brew install python
    brew install --build-from-source boost
    brew install snappy leveldb protobuf gflags glog
    brew tap homebrew/science
    brew install homebrew/science/hdf5
    brew install homebrew/science/opencv

Building boost from source is needed to link against your local python.
If using homebrew python, python packages like `numpy` and `scipy` are best installed by doing `brew tap homebrew/python`, and then installing them with homebrew.

#### 10.9 additional notes

In OS X 10.9 Apple changed to clang as the default compiler. Clang uses libc++ as the standard library by default, while NVIDIA CUDA currently works with libstdc++. This makes it necessary to change the compilation settings for each of the dependencies. We do this by modifying the homebrew formulae before installing any packages. Make sure that homebrew doesn't install any software dependencies in the background; all packages must be linked to libstdc++.

Only Anaconda python has been confirmed to work on 10.9.

For each package that you install through homebrew do the following:

1. Open formula in editor: `brew edit FORMULA`
2. Add the ENV definitions as shown in the code block below.
3. Uninstall any formulae that were already installed: `brew uninstall FORMULA`
4. Install / Reinstall: `brew install --build-from-source --fresh -vd FORMULA`

```
    def install
        #ADD THE FOLLOWING:
        ENV.append "CXXFLAGS", '-stdlib=libstdc++'
        ENV.append "CFLAGS", '-stdlib=libstdc++'
        ENV.append "LDFLAGS", '-stdlib=libstdc++ -lstdc++'
        #The following is necessary because libtool liks to strip LDFLAGS:
        ENV.cxx = "/usr/bin/clang -stdlib=libstdc++"

        ...
```

The prerequisite homebrew formulae are

    cmake boost snappy leveldb protobuf gflags glog homebrew/science/opencv

so follow steps 1-4 for each.

After this the rest of the installation is the same as under 10.8, as long as `clang++` is invoked with `-stdlib=libstdc++` and `-lstdc++` is linked.

### CUDA and MKL

CUDA and MKL are very straightforward to install; download from the NVIDIA and Intel websites.

### Compiling Caffe

Here are the relevant parts of the Makefile.config after all this:

    CUDA_DIR := /Developer/NVIDIA/CUDA-5.5
    MKL_DIR := /opt/intel/mkl
    PYTHON_INCLUDES := /path/to/anaconda/include /path/to/anaconda/include/python2.7 /path/to/anaconda/lib/python2.7/site-packages/numpy/core/include
    PYTHON_LIB := /path/to/anaconda/lib
    CXX=/usr/bin/clang++

Don't forget to set `PATH` and `LD_LIBRARY_PATH`:

    export PATH=/path/to/anaconda/bin:/Developer/NVIDIA/CUDA-5.5/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/X11/bin
    export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-5.5/lib:/opt/intel/composer_xe_2013_sp1.1.103/compiler/lib:/opt/intel/composer_xe_2013_sp1.1.103/mkl/lib:/path/to/anaconda/lib:/usr/local/lib:/usr/lib:/lib

Additionally, MKL requires `DYLD_LIBRARY_PATH` to be set:

    export MKL_DIR=/opt/intel/composer_xe_2013_sp1.1.103
    export DYLD_LIBRARY_PATH=$MKL_DIR/compiler/lib:$MKL_DIR/mkl/lib

Note that we still need to include the MKL `compiler/lib` in our paths, although we do not explicitly link against this directory in the Makefile.

Further note that these paths are for Anaconda python. For homebrew python, substitute `/usr/local/Cellar/python/2.7.6/Frameworks/Python.framework/Versions/2.7` for `/path/to/anaconda`.

## Hardware Questions

**Laboratory Tested Hardware**: Berkeley Vision runs Caffe with k40s, k20s, and Titans including models at ImageNet/ILSVRC scale. We also run on GTX series cards and GPU-equipped MacBook Pros. We have not encountered any trouble in-house with devices with CUDA capability >= 3.0. All reported hardware issues thus-far have been due to GPU configuration, overheating, and the like.

**CUDA compute capability**: devices with compute capability <= 2.0 may have to reduce CUDA thread numbers and batch sizes due to hardware constraints. Your mileage may vary.

Refer to the project's issue tracker for [hardware/compatibility](https://github.com/BVLC/caffe/issues?labels=hardware%2Fcompatibility&page=1&state=open).
