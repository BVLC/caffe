
---
title: Installation
---

# Installation

Prior to installing, have a glance through this guide and take note of the details for your platform.
We install and run Caffe on Ubuntu 14.04, CentOS (7.0, 7.1, 7.2), and AWS.
The official Makefile and `Makefile.config` build are complemented by an automatic CMake build from the community.

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Hardware](#hardware)

When updating Caffe, it's best to `make clean` before re-compiling.

## Prerequisites

Before building Caffe make sure that the following dependencies are available on target system:

* [BLAS library](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
    * [Intel® Math Kernel Library (Intel &reg; MKL)](https://software.intel.com/en-us/intel-mkl)
    * [Open BLAS](http://www.openblas.net)
    * [ATLAS](http://math-atlas.sourceforge.net)
* [Boost](http://www.boost.org/) >= 1.55
* `protobuf`, `glog`, `gflags`, `hdf5`

For additional capabilities and acceleration the following dependencies might be necessary:

* [OpenCV](http://opencv.org/) >= 2.4 including 3.0
* IO libraries: `lmdb`, `leveldb` (note: leveldb requires `snappy`)
* For GPU mode
    * [CUDA](https://developer.nvidia.com/cuda-zone)
    * cuDNN

* For Pycaffe
    * `Python 2.7` or `Python 3.3+`
    * `numpy (>= 1.7)`
    * boost-provided `boost.python`

* For Matcaffe 
  * MATLAB with the `mex` compiler.

### Building for Intel® Architecture

This version of Caffe is optimized for Intel® Xeon processors and Intel® Xeon Phi™ processors. To achieve the best performance results on Intel Architecture we recommend building Caffe with [Intel MKL](http://software.intel.com/en-us/intel-mkl) and enabling OpenMP support. If you don't have Intel MKL yet you can download it [free of charge](https://software.intel.com/en-us/articles/free_mkl). The following configuration changes are recommended:

* Set `BLAS := mkl` in `Makefile.config`
* If you don't need GPU optimizations `CPU_ONLY := 1` flag in `Makefile.config` to configure and build Caffe without CUDA.

[Intel MKL 2017 Beta Update 1](https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/623305) introduces optimized Deep Neural Network (DNN) performance primitives that allow to accelerate the most popular image recognition topologies. Caffe can take advantage of these primitives and get significantly better performance results compared to the previous versions of Intel MKL. There are two ways to take advantage of the new primitives: 

* At Caffe build time add `USE_MKL2017_AS_DEFAULT_ENGINE := 1` to `Makefile.config` or add `-DUSE_MKL2017_AS_DEFAULT_ENGINE=ON` to your commandline when invoking `cmake`. All layers will use new primitives by default.
* Set layer engine to `MKL2017` in model configuration. Only this specific layer will be accelerated with new primitives. 

#### Recommendations
* For Better performance please disable Hyperthreading on your platoform.

### Building for GPU
Caffe requires the CUDA `nvcc` compiler to compile its GPU code and CUDA driver for GPU operation.
To install CUDA, go to the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. Install the library and the latest standalone driver separately; the driver bundled with the library is usually out-of-date. **Warning!** The 331.* CUDA driver series has a critical performance issue: do not use it.

For best performance on GPU, Caffe can be accelerated by [NVIDIA cuDNN](https://developer.nvidia.com/cudnn). Register for free at the cuDNN site, install it, then continue with these installation instructions. To compile with cuDNN set the `USE_CUDNN := 1` flag set in your `Makefile.config`.

Caffe requires BLAS as the backend of its matrix and vector computations. There are several implementations of this library. The choice is yours:

* [ATLAS](http://math-atlas.sourceforge.net/): free, open source, and so the default for Caffe.
* [Intel MKL](http://software.intel.com/en-us/intel-mkl): free performance library for Intel Architecture
    1. Install Intel MKL. Free options [are available](https://software.intel.com/en-us/articles/free_mkl)
    2. Set `BLAS := mkl` in `Makefile.config`
* [OpenBLAS](http://www.openblas.net/): free and open source; this optimized and parallel BLAS could require more effort to install, although it might offer a speedup.
    1. Install OpenBLAS
    2. Set `BLAS := open` in `Makefile.config`

### Python and/or MATLAB Caffe (optional)

#### Python

The main requirements are `numpy` and `boost.python` (provided by boost). `pandas` is useful too and needed for some examples.

You can install the dependencies with

    for req in $(cat requirements.txt); do pip install $req; done

but we suggest first installing the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which provides most of the necessary packages, as well as the `hdf5` library dependency.

To import the `caffe` Python module after completing the installation, add the module directory to your `$PYTHONPATH` by `export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH` or the like. You should not import the module in the `caffe/python/caffe` directory!

*Caffe's Python interface works with Python 2.7. Python 3.3+ should work out of the box without protobuf support. For protobuf support please install protobuf 3.0 alpha (https://developers.google.com/protocol-buffers/). Earlier Pythons are your own adventure.*

#### MATLAB

Install MATLAB, and make sure that its `mex` is in your `$PATH`.

*Caffe's MATLAB interface works with versions 2015a, 2014a/b, 2013a/b, and 2012b.*

## Compilation

Caffe can be compiled with either Make or CMake. Make is officially supported while CMake is supported by the community.

### Compilation with Make

Configure the build by copying and modifying the example `Makefile.config` for your setup. The defaults should work, but uncomment the relevant lines if using Anaconda Python.

    cp Makefile.config.example Makefile.config
    # Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
    make all
    make test
    make runtest

- For CPU & GPU accelerated Caffe, no changes are needed.
- For cuDNN acceleration using NVIDIA's proprietary cuDNN software, uncomment the `USE_CUDNN := 1` switch in `Makefile.config`. cuDNN is sometimes but not always faster than Caffe's GPU acceleration.
- For CPU-only Caffe, uncomment `CPU_ONLY := 1` in `Makefile.config`.

To compile the Python and MATLAB wrappers do `make pycaffe` and `make matcaffe` respectively.
Be sure to set your MATLAB and Python paths in `Makefile.config` first!

**Distribution**: run `make distribute` to create a `distribute` directory with all the Caffe headers, compiled libraries, binaries, etc. needed for distribution to other machines.

**Speed**: for a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

Now that you have installed Caffe, check out the [MNIST tutorial](gathered/examples/mnist.html) and the [reference ImageNet model tutorial](gathered/examples/imagenet.html).

### CMake Build

In lieu of manually editing `Makefile.config` to configure the build, Caffe offers an unofficial CMake build thanks to @Nerei, @akosiorek, and other members of the community. It requires CMake version >= 2.8.7.
The basic steps are as follows:

    mkdir build
    cd build
    cmake ..
    make all
    make install
    make runtest

See [PR #1667](https://github.com/BVLC/caffe/pull/1667) for options and details.

## Hardware
### Intel Architecture
This software supports the following hardware:
* Intel® Xeon processor E5-xxxx v3 (codename Haswell) and Intel® Xeon processor E5-xxxx v4 (codename Broadwell)
* Next generation Intel® Xeon Phi™ product family (codenamed Knights Landing)

### GPU
Berkeley Vision runs Caffe with K40s, K20s, and Titans including models at ImageNet/ILSVRC scale. We also run on GTX series cards (980s and 770s) and GPU-equipped MacBook Pros. We have not encountered any trouble in-house with devices with CUDA capability >= 3.0. All reported hardware issues thus-far have been due to GPU configuration, overheating, and the like.

Once installed, check your times against our [reference performance numbers](performance_hardware.html) to make sure everything is configured properly.

Ask hardware questions on the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users).
