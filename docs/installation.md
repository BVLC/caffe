---
layout: default
title: Caffe
---

# Installation

Prior to installing, it is best to read through this guide and take note of the details for your platform.
We have successfully compiled and run Caffe on Ubuntu 12.04, OS X 10.8, and OS X 10.9.

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Hardware questions](#hardware_questions)

## Prerequisites

Caffe depends on several software packages.

* [CUDA](https://developer.nvidia.com/cuda-zone) (5.0, 5.5, or 6.0).
* [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (provided via ATLAS, MKL, or OpenBLAS).
* [OpenCV](http://opencv.org/).
* [Boost](http://www.boost.org/) (we have only tested 1.55)
* `glog`, `gflags`, `protobuf`, `leveldb`, `snappy`, `hdf5`
* For the python wrapper
    * `python`, `numpy (>= 1.7)`, Boost-provided `boost.python`
* For the MATLAB wrapper
    * MATLAB with the `mex` compiler.

### CUDA and BLAS

Caffe requires the CUDA `nvcc` compiler to compile its GPU code.
To install CUDA, go to the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. **Note:** you can install the CUDA libraries without a CUDA card or driver, in order to build and run Caffe on a CPU-only machine.

Caffe requires BLAS as the backend of its matrix and vector computations.
There are several implementations of this library.
The choice is yours:

* [ATLAS](http://math-atlas.sourceforge.net/): free, open source, and so the default for Caffe.
    + Ubuntu: `sudo apt-get install libatlas-base-dev`
    + CentOS/RHEL: `sudo yum install libatlas-devel`
    + OS X: already installed as the [Accelerate / vecLib Framework](https://developer.apple.com/library/mac/documentation/Darwin/Reference/ManPages/man7/Accelerate.7.html).
* [Intel MKL](http://software.intel.com/en-us/intel-mkl): commercial and optimized for Intel CPUs, with a free trial and [student](http://software.intel.com/en-us/intel-education-offerings) licenses.
    1. Install MKL.
    2. Set `BLAS := mkl` in `Makefile.config`
* [OpenBLAS](http://www.openblas.net/): free and open source; this optimized and parallel BLAS could require more effort to install, although it might offer a speedup.
    1. Install OpenBLAS
    2. Set `BLAS := open` in `Makefile.config`

### Python and/or Matlab wrappers (optional)

This is only a requirement if you'd like the Python wrapper for Caffe.
The main required package is `numpy`, and `Boost` (in "Other dependencies" below) must be compiled with Python support.

For **OS X**, we highly recommend using the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which provides most of the necessary packages, as well as the `hdf5` library dependency.
If you don't, please use Homebrew -- but beware of potential linking errors!

Note that if you use the **Ubuntu** default python, you will need to `apt-get install` the `python-dev` package to have the python headers. You can install any remaining dependencies with

    pip install -r /path/to/caffe/python/requirements.txt

If you would like to have the MATLAB wrapper, install MATLAB, and make sure that its `mex` is in your `$PATH`.

### The rest of the dependencies

#### Linux

On **Ubuntu**, the remaining dependencies can be installed with

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

And on **CentOS or RHEL**, you can install via yum using:

    sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

The only exception being the google logging library, which does not exist in the Ubuntu 12.04 or CentOS/RHEL repositories. To install it, do:

    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    ./configure
    make && make install

#### OS X

On **OS X**, we highly recommend using the [homebrew](http://brew.sh/) package manager, and ideally starting from a clean install of the OS (or from a wiped `/usr/local`) to avoid conflicts.
In the following, we assume that you're using Anaconda Python and Homebrew.

To install the OpenCV dependency, we'll need to provide an additional source for Homebrew:

    brew tap homebrew/science

If using Anaconda Python, a modification is required to the OpenCV formula.
Do `brew edit opencv` and change the lines that look like the two lines below to exactly the two lines below.

      -DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib
      -DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7

**NOTE**: We find that everything compiles successfully if `$LD_LIBRARY_PATH` is not set at all, and `$DYLD_FALLBACK_LIBRARY_PATH` is set to to provide CUDA, Python, and other relevant libraries (e.g. `/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib`).
In other `ENV` settings, things may not work as expected.

#### 10.8-specific Instructions

Simply run the following.

    brew install --build-from-source boost
    for x in snappy leveldb protobuf gflags glog szip homebrew/science/opencv; do brew install $x; done

Building boost from source is needed to link against your local python (exceptions might be raised during some OS X installs, but **ignore** these and continue).

**Note** that the HDF5 dependency is provided by Anaconda Python in this case.
If you're not using Anaconda, include `hdf5` in the list above.

#### 10.9-specific Instructions

In OS X 10.9, clang++ is the default C++ compiler and uses `libc++` as the standard library.
However, NVIDIA CUDA (even version 6.0) currently links only with `libstdc++`.
This makes it necessary to change the compilation settings for each of the dependencies.

We do this by modifying the homebrew formulae before installing any packages.
Make sure that homebrew doesn't install any software dependencies in the background; all packages must be linked to `libstdc++`.

The prerequisite homebrew formulae are

    boost snappy leveldb protobuf gflags glog szip homebrew/science/opencv

For each of these formulas, `brew edit FORMULA`, and add the ENV definitions as shown:

      def install
          # ADD THE FOLLOWING:
          ENV.append "CXXFLAGS", "-stdlib=libstdc++"
          ENV.append "CFLAGS", "-stdlib=libstdc++"
          ENV.append "LDFLAGS", "-stdlib=libstdc++ -lstdc++"
          # The following is necessary because libtool likes to strip LDFLAGS:
          ENV["CXX"] = "/usr/bin/clang++ -stdlib=libstdc++"
          ...

To edit the formulae in turn, run

    for x in snappy leveldb protobuf gflags glog szip boost homebrew/science/opencv; do brew edit $x; done

After this, run

    for x in snappy leveldb protobuf gflags glog szip boost homebrew/science/opencv; do brew uninstall $x; brew install --build-from-source --fresh -vd $x; done

**Note** that the HDF5 dependency is provided by Anaconda Python in this case.
If you're not using Anaconda, include `hdf5` in the list above.

#### Windows

There is an unofficial Windows port of Caffe at [niuzhiheng/caffe:windows](https://github.com/niuzhiheng/caffe). Thanks [@niuzhiheng](https://github.com/niuzhiheng).

## Compilation

Now that you have the prerequisites, edit your `Makefile.config` to change the paths for your setup.
The defaults should work, but uncomment the relevant lines if using Anaconda Python.

    cp Makefile.config.example Makefile.config
    # Adjust Makefile.config (for example, if using Anaconda Python)
    make all
    make test
    make runtest

Note that if there is no GPU in your machine, building and running CPU-only works, but GPU tests will naturally fail.

To compile the python and MATLAB wrappers do `make pycaffe` and `make matcaffe` respectively.
Be sure to set your MATLAB and python paths in `Makefile.config` first!
For Python support, you must add the compiled module to your `PYTHONPATH` (as `/path/to/caffe/python` or the like).

*Distribution*: run `make distribute` to create a `distribute` directory with all the Caffe headers, compiled libraries, binaries, etc. needed for distribution to other machines.

*Speed*: for a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

Now that you have installed Caffe, check out the [MNIST demo](mnist.html) and the pretrained [ImageNet example](imagenet.html).

## Hardware Questions

**Laboratory Tested Hardware**: Berkeley Vision runs Caffe with K40s, K20s, and Titans including models at ImageNet/ILSVRC scale. We also run on GTX series cards and GPU-equipped MacBook Pros. We have not encountered any trouble in-house with devices with CUDA capability >= 3.0. All reported hardware issues thus-far have been due to GPU configuration, overheating, and the like.

**CUDA compute capability**: devices with compute capability <= 2.0 may have to reduce CUDA thread numbers and batch sizes due to hardware constraints. Your mileage may vary.

Refer to the project's issue tracker for [hardware/compatibility](https://github.com/BVLC/caffe/issues?labels=hardware%2Fcompatibility&page=1&state=open).
