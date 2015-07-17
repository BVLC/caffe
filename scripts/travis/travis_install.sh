#!/bin/bash
# This script must be run with sudo.

set -e

MAKE="make --jobs=$NUM_THREADS"

# Install apt packages where the Ubuntu 12.04 default and ppa works for Caffe

# This ppa is for gflags and glog
add-apt-repository -y ppa:tuleu/precise-backports
# This ppa is for boost 1.54
add-apt-repository -y ppa:boost-latest/ppa
# This ppa is for g++ 4.8
add-apt-repository -y ppa:ubuntu-toolchain-r/test

apt-get -y update
apt-get install \
    g++-4.8 wget git curl \
    python-dev python-numpy \
    libleveldb-dev libsnappy-dev libopencv-dev \
    libboost-python1.54-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev libgflags-dev libgoogle-glog-dev \
    bc

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 90
# Add a special apt-repository to install CMake 2.8.9 for CMake Caffe build,
# if needed.  By default, Aptitude in Ubuntu 12.04 installs CMake 2.8.7, but
# Caffe requires a minimum CMake version of 2.8.8.
if $WITH_CMAKE; then
  add-apt-repository -y ppa:ubuntu-sdk-team/ppa
  apt-get -y update
  apt-get -y install cmake
fi

# Install CUDA, if needed
if $WITH_CUDA; then
  CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_7.0-28_amd64.deb
  CUDA_FILE=/tmp/cuda_install.deb
  curl $CUDA_URL -o $CUDA_FILE
  dpkg -i $CUDA_FILE
  rm -f $CUDA_FILE
  apt-get -y update
  # Install the minimal CUDA subpackages required to test Caffe build.
  # For a full CUDA installation, add 'cuda' to the list of packages.
  apt-get -y install cuda-core-7-0 cuda-cublas-7-0 cuda-cublas-dev-7-0 cuda-cudart-7-0 cuda-cudart-dev-7-0 cuda-curand-7-0 cuda-curand-dev-7-0
  # Create CUDA symlink at /usr/local/cuda
  # (This would normally be created by the CUDA installer, but we create it
  # manually since we did a partial installation.)
  ln -s /usr/local/cuda-7.0 /usr/local/cuda
fi

# Install LMDB
LMDB_URL=https://github.com/LMDB/lmdb/archive/LMDB_0.9.14.tar.gz
LMDB_FILE=/tmp/lmdb.tar.gz
pushd .
wget $LMDB_URL -O $LMDB_FILE
tar -C /tmp -xzvf $LMDB_FILE
cd /tmp/lmdb*/libraries/liblmdb/
$MAKE
$MAKE install
popd
rm -f $LMDB_FILE

# Install the Python runtime dependencies via miniconda (this is much faster
# than using pip for everything).
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf
