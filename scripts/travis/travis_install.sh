#!/bin/bash

# Install apt packages where the Ubuntu 12.04 default works for Caffe
sudo apt-get -y update
sudo apt-get install \
    wget git curl \
    python-dev python-numpy \
    libleveldb-dev libsnappy-dev libopencv-dev \
    libboost-dev libboost-system-dev libboost-python-dev libboost-thread-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev \
    bc

# Add a special apt-repository to install CMake 2.8.9 for CMake Caffe build,
# if needed.  By default, Aptitude in Ubuntu 12.04 installs CMake 2.8.7, but
# Caffe requires a minimum CMake version of 2.8.8.
if $WITH_CMAKE; then
    sudo add-apt-repository -y ppa:ubuntu-sdk-team/ppa
    sudo apt-get -y update
    sudo apt-get -y install cmake || (echo "CMake install failed"; exit 1)
fi

# Install glog
GLOG_URL=https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
GLOG_FILE=/tmp/glog-0.3.3.tar.gz
pushd .
wget $GLOG_URL -O $GLOG_FILE && \
    tar -C /tmp -xzvf $GLOG_FILE && \
    rm $GLOG_FILE && \
    cd /tmp/glog-0.3.3 && \
    ./configure && make && sudo make install || \
    (echo "glog install failed"; exit 1)
popd

# Install gflags
GFLAGS_URL=https://github.com/schuhschuh/gflags/archive/master.zip
GFLAGS_FILE=/tmp/gflags-master.zip
pushd .
wget $GFLAGS_URL -O $GFLAGS_FILE && \
    cd /tmp/ && unzip gflags-master.zip && \
    cd gflags-master && \
    mkdir build && \
    cd build && \
    export CXXFLAGS="-fPIC" && \
    cmake .. && make VERBOSE=1 && sudo make install || \
    (echo "gflags install failed"; exit 1)
popd

# Install CUDA, if needed
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_6.0-37_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
if $WITH_CUDA; then
    curl $CUDA_URL -o $CUDA_FILE && \
        sudo dpkg -i $CUDA_FILE ||
        (echo "CUDA install failed"; exit 1)
    rm -f $CUDA_FILE
    sudo apt-get -y update
    # Install the minimal CUDA subpackages required to test Caffe build.
    # For a full CUDA installation, add 'cuda' to the list of packages.
    sudo apt-get -y install cuda-core-6-0 cuda-extra-libs-6-0
    # Create CUDA symlink at /usr/local/cuda
    # (This would normally be created by the CUDA installer, but we create it
    # manually since we did a partial installation.)
    sudo ln -s /usr/local/cuda-6.0 /usr/local/cuda ||
        (echo "CUDA symlink creation failed"; exit 1)
fi

# Install LMDB
LMDB_URL=https://gitorious.org/mdb/mdb/archive/7f038d0f15bec57b4c07aa3f31cd5564c88a1897.tar.gz
LMDB_FILE=/tmp/mdb.tar.gz
pushd .
curl $LMDB_URL -o $LMDB_FILE && \
    tar -C /tmp -xzvf $LMDB_FILE && \
    cd /tmp/mdb-mdb/libraries/liblmdb/ && \
    make && sudo make install || \
    (echo "LMDB install failed"; exit 1)
popd
rm -f $LMDB_FILE
