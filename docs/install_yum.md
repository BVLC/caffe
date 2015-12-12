---
title: Installation: RHEL / Fedora / CentOS
---

# RHEL / Fedora / CentOS Installation

**General dependencies**

    sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel

**Remaining dependencies, recent OS**

    sudo yum install gflags-devel glog-devel lmdb-devel

**Remaining dependencies, if not found**

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
    git clone https://github.com/LMDB/lmdb
    cd lmdb/libraries/liblmdb
    make && make install

Note that glog does not compile with the most recent gflags version (2.1), so before that is resolved you will need to build with glog first.

**CUDA**: Install via the NVIDIA package instead of `yum` to be certain of the library and driver versions.
Install the library and latest driver separately; the driver bundled with the library is usually out-of-date.
    + CentOS/RHEL/Fedora:

**BLAS**: install ATLAS by `sudo yum install atlas-devel` or install OpenBLAS or MKL for better CPU performance. For the Makefile build, uncomment and set `BLAS_LIB` accordingly as ATLAS is usually installed under `/usr/lib[64]/atlas`).

**Python** (optional): if you use the default Python you will need to `sudo yum install` the `python-devel` package to have the Python headers for building the pycaffe wrapper.

Continue with [compilation](installation.html#compilation).
