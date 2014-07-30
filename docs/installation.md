---
layout: default
title: Caffe
---

# Installation

Prior to installing, it is best to read through this guide and take note of the details for your platform.
We have installed Caffe on Ubuntu 14.04, Ubuntu 12.04, OS X 10.9, and OS X 10.8.

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Hardware questions](#hardware_questions)

## Prerequisites

Caffe depends on several software packages.

* [CUDA](https://developer.nvidia.com/cuda-zone) library version 6.0, 5.5, or 5.0 and the latest driver version for CUDA 6 or 319.* for CUDA 5 (and NOT 331.*)
* [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (provided via ATLAS, MKL, or OpenBLAS).
* [OpenCV](http://opencv.org/).
* [Boost](http://www.boost.org/) (>= 1.55, although only 1.55 is tested)
* `glog`, `gflags`, `protobuf`, `leveldb`, `snappy`, `hdf5`, `lmdb`
* For the Python wrapper
    * `Python 2.7`, `numpy (>= 1.7)`, boost-provided `boost.python`
* For the MATLAB wrapper
    * MATLAB with the `mex` compiler.

**CPU-only Caffe**: for cold-brewed CPU-only Caffe uncomment the `CPU_ONLY := 1` in `Makefile.config` to configure and build Caffe without CUDA. This is helpful for cloud or cluster deployment.

### CUDA and BLAS

Caffe requires the CUDA `nvcc` compiler to compile its GPU code and CUDA driver for GPU operation.
To install CUDA, go to the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads) and follow installation instructions there. Install the library and the latest standalone driver separately; the driver bundled with the library is usually out-of-date. **Warning!** The 331.* CUDA driver series has a critical performance issue: do not use it.

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

### Python and/or MATLAB wrappers (optional)

#### Python

The main requirements are `numpy` and `boost.python` (provided by boost). `pandas` is useful too and needed for some examples.

You can install the dependencies with

    pip install -r /path/to/caffe/python/requirements.txt

but we highly recommend first installing the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which provides most of the necessary packages, as well as the `hdf5` library dependency.

For **Ubuntu**, if you use the default Python you will need to `apt-get install` the `python-dev` package to have the Python headers for building the wrapper.

For **OS X**, Anaconda is the preferred Python. If you decide against it, please use Homebrew -- but beware of potential linking errors!

To import the `caffe` Python module after completing the installation, add the module directory to your `$PYTHONPATH` by `export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH` or the like. You should not import the module in the `caffe/python/caffe` directory!

*Caffe's Python interface works with Python 2.7. Python 3 or earlier Pythons are your own adventure.*

#### MATLAB

Install MATLAB, and make sure that its `mex` is in your `$PATH`.

*Caffe's MATLAB interface works with versions 2012b, 2013a/b, and 2014a.*

### The rest of the dependencies

#### Linux

On **Ubuntu**, most of the dependencies can be installed with

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

And on **CentOS / RHEL**, you can install via yum with

    sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

and for **Ubuntu 14.04** the rest of the dependencies can be installed with

    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

For **Ubuntu 12.04 and CentOS / RHEL** the only exceptions to package installation are the Google flags library, Google logging library, and LMDB. To install these, do:

    # glog
    wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
    tar zxvf glog-0.3.3.tar.gz
    cd glog-0.3.3
    ./configure
    make && make install
    # gflags
    wget https://github.com/schuhschuh/gflags/archive/master.zip
    unzip master.zip
    cd gflags-master
    mkdir build && cd build
    export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
    make && make install
    # lmdb
    git clone git://gitorious.org/mdb/mdb.git
    cd mdb/libraries/liblmdb
    make && make install

Note that glog does not compile with the most recent gflags version (2.1), so before that is resolved you will need to build with glog first.

#### OS X

On **OS X**, we highly recommend using the [Homebrew](http://brew.sh/) package manager, and ideally starting from a clean install of the OS (or from a wiped `/usr/local`) to avoid conflicts.
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

Simply run the following:

    brew install --build-from-source --with-python boost
    for x in snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv; do brew install $x; done

Building boost from source is needed to link against your local Python (exceptions might be raised during some OS X installs, but **ignore** these and continue). If you do not need the Python wrapper, simply doing `brew install boost` is fine.

**Note** that the HDF5 dependency is provided by Anaconda Python in this case.
If you're not using Anaconda, include `hdf5` in the list above.

#### 10.9-specific Instructions

In OS X 10.9, clang++ is the default C++ compiler and uses `libc++` as the standard library.
However, NVIDIA CUDA (even version 6.0) currently links only with `libstdc++`.
This makes it necessary to change the compilation settings for each of the dependencies.

We do this by modifying the Homebrew formulae before installing any packages.
Make sure that Homebrew doesn't install any software dependencies in the background; all packages must be linked to `libstdc++`.

The prerequisite Homebrew formulae are

    boost snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv

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

    for x in snappy leveldb protobuf gflags glog szip boost lmdb homebrew/science/opencv; do brew edit $x; done

After this, run

    for x in snappy leveldb protobuf gflags glog szip lmdb homebrew/science/opencv; do brew uninstall $x; brew install --build-from-source --fresh -vd $x; done
    brew install --build-from-source --with-python --fresh -vd boost

**Note** that `brew install --build-from-source --fresh -vd boost` is fine if you do not need the Caffe Python wrapper.

**Note** that the HDF5 dependency is provided by Anaconda Python in this case.
If you're not using Anaconda, include `hdf5` in the list above.

**Note** that in order to build the caffe python wrappers you must install boost using the --with-python option:

    brew install --build-from-source --with-python --fresh -vd boost

#### Windows

There is an unofficial Windows port of Caffe at [niuzhiheng/caffe:windows](https://github.com/niuzhiheng/caffe). Thanks [@niuzhiheng](https://github.com/niuzhiheng)!

## Compilation

Now that you have the prerequisites, edit your `Makefile.config` to change the paths for your setup.
The defaults should work, but uncomment the relevant lines if using Anaconda Python.

    cp Makefile.config.example Makefile.config
    # Adjust Makefile.config (for example, if using Anaconda Python)
    make all
    make test
    make runtest

If there is no GPU in your machine, you should switch to CPU-only Caffe by uncommenting `CPU_ONLY := 1` in `Makefile.config`.

To compile the Python and MATLAB wrappers do `make pycaffe` and `make matcaffe` respectively.
Be sure to set your MATLAB and Python paths in `Makefile.config` first!

*Distribution*: run `make distribute` to create a `distribute` directory with all the Caffe headers, compiled libraries, binaries, etc. needed for distribution to other machines.

*Speed*: for a faster build, compile in parallel by doing `make all -j8` where 8 is the number of parallel threads for compilation (a good choice for the number of threads is the number of cores in your machine).

Now that you have installed Caffe, check out the [MNIST demo](mnist.html) and the pretrained [ImageNet example](imagenet.html).

## Hardware Questions

**Laboratory Tested Hardware**: Berkeley Vision runs Caffe with K40s, K20s, and Titans including models at ImageNet/ILSVRC scale. We also run on GTX series cards and GPU-equipped MacBook Pros. We have not encountered any trouble in-house with devices with CUDA capability >= 3.0. All reported hardware issues thus-far have been due to GPU configuration, overheating, and the like.

**CUDA compute capability**: devices with compute capability <= 2.0 may have to reduce CUDA thread numbers and batch sizes due to hardware constraints. Your mileage may vary.

Once installed, check your times against our [reference performance numbers](performance_hardware.html) to make sure everything is configured properly.

Refer to the project's issue tracker for [hardware/compatibility](https://github.com/BVLC/caffe/issues?labels=hardware%2Fcompatibility&page=1&state=open).
