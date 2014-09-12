#!/bin/bash
# This script must be run with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"

# Install apt packages where the Ubuntu 12.04 default works for Caffe
apt-get -y update
apt-get install \
    wget git curl \
    python-dev python-numpy \
    libleveldb-dev libsnappy-dev libopencv-dev \
    libboost-dev libboost-system-dev libboost-filesystem-dev libboost-python-dev libboost-thread-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev \
    bc

# Add a special apt-repository to install CMake 2.8.9 for CMake Caffe build,
# if needed.  By default, Aptitude in Ubuntu 12.04 installs CMake 2.8.7, but
# Caffe requires a minimum CMake version of 2.8.8.
if $WITH_CMAKE; then
  add-apt-repository -y ppa:ubuntu-sdk-team/ppa
  apt-get -y update
  apt-get -y install cmake
fi

# Install glog
GLOG_URL=https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
GLOG_FILE=/tmp/glog-0.3.3.tar.gz
pushd .
wget $GLOG_URL -O $GLOG_FILE
tar -C /tmp -xzvf $GLOG_FILE
rm $GLOG_FILE
cd /tmp/glog-0.3.3
./configure
$MAKE
$MAKE install
popd

# Install gflags
GFLAGS_URL=https://github.com/schuhschuh/gflags/archive/master.zip
GFLAGS_FILE=/tmp/gflags-master.zip
pushd .
wget $GFLAGS_URL -O $GFLAGS_FILE
cd /tmp/
unzip gflags-master.zip
cd gflags-master
mkdir build
cd build
export CXXFLAGS="-fPIC"
cmake ..
$MAKE VERBOSE=1
$MAKE install
popd

# Install CUDA, if needed
if $WITH_CUDA; then
  CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_6.0-37_amd64.deb
  CUDA_FILE=/tmp/cuda_install.deb
  curl $CUDA_URL -o $CUDA_FILE
  dpkg -i $CUDA_FILE
  rm -f $CUDA_FILE
  apt-get -y update
  # Install the minimal CUDA subpackages required to test Caffe build.
  # For a full CUDA installation, add 'cuda' to the list of packages.
  apt-get -y install cuda-core-6-0 cuda-extra-libs-6-0
  # Create CUDA symlink at /usr/local/cuda
  # (This would normally be created by the CUDA installer, but we create it
  # manually since we did a partial installation.)
  ln -s /usr/local/cuda-6.0 /usr/local/cuda
fi

# Install LMDB
LMDB_URL=ftp://ftp.openldap.org/pub/OpenLDAP/openldap-release/openldap-2.4.39.tgz
LMDB_FILE=/tmp/openldap.tgz
pushd .
curl $LMDB_URL -o $LMDB_FILE
tar -C /tmp -xzvf $LMDB_FILE
cd /tmp/openldap*/libraries/liblmdb/
$MAKE
$MAKE install
popd
rm -f $LMDB_FILE
