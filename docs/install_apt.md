---
title: Installation: Ubuntu
---

# Ubuntu Installation

**General dependencies**

    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
    sudo apt-get install --no-install-recommends libboost-all-dev

**CUDA**: Install via the NVIDIA package instead of `apt-get` to be certain of the library and driver versions.
Install the library and latest driver separately; the driver bundled with the library is usually out-of-date.
This can be skipped for CPU-only installation.

**BLAS**: install ATLAS by `sudo apt-get install libatlas-base-dev` or install OpenBLAS or MKL for better CPU performance.

**Python** (optional): if you use the default Python you will need to `sudo apt-get install` the `python-dev` package to have the Python headers for building the pycaffe interface.

**MASS** (optional): Install MASS library as below -

    wget -q http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/public.gpg -O- | sudo apt-key add -
    echo "deb http://public.dhe.ibm.com/software/server/POWER/Linux/xl-compiler/eval/ppc64le/ubuntu/ trusty main" | sudo tee /etc/apt/sources.list.d/ibm-xl-compiler-eval.list
    sudo apt-get update
    sudo apt-get install libxlmass-devel.8.1.3

**Remaining dependencies, 14.04**

Everything is packaged in 14.04.

    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

**Remaining dependencies, 12.04**

These dependencies need manual installation in 12.04.

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

Continue with [compilation](installation.html#compilation).
