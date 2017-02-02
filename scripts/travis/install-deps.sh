#!/bin/bash
# install dependencies
# (this script must be run as root)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

apt-get -y update
apt-get install -y --no-install-recommends \
  build-essential \
  libboost-filesystem-dev \
  libboost-python-dev \
  libboost-system-dev \
  libboost-thread-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libhdf5-serial-dev \
  libopenblas-dev \
  python-virtualenv \
  wget

if $WITH_CMAKE ; then
  apt-get install -y --no-install-recommends cmake
fi

if ! $WITH_PYTHON3 ; then
  # Python2
  apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-protobuf \
    python-skimage
else
  # Python3
  apt-get install -y --no-install-recommends \
    python3-dev \
    python3-numpy \
    python3-skimage

  # build Protobuf3 since it's needed for Python3
  PROTOBUF3_DIR=~/protobuf3
  pushd .
  if [ -d "$PROTOBUF3_DIR" ] && [ -e "$PROTOBUF3_DIR/src/protoc" ]; then
    echo "Using cached protobuf3 build ..."
    cd $PROTOBUF3_DIR
  else
    echo "Building protobuf3 from source ..."
    rm -rf $PROTOBUF3_DIR
    mkdir $PROTOBUF3_DIR

    # install some more dependencies required to build protobuf3
    apt-get install -y --no-install-recommends \
      curl \
      dh-autoreconf \
      unzip

    wget https://github.com/google/protobuf/archive/3.0.x.tar.gz -O protobuf3.tar.gz
    tar -xzf protobuf3.tar.gz -C $PROTOBUF3_DIR --strip 1
    rm protobuf3.tar.gz
    cd $PROTOBUF3_DIR
    ./autogen.sh
    ./configure --prefix=/usr
    make --jobs=$NUM_THREADS
  fi
  make install
  popd
fi

if $WITH_IO ; then
  apt-get install -y --no-install-recommends \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libsnappy-dev
fi

if $WITH_CUDA ; then
  # install repo packages
  CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG

  if $WITH_CUDNN ; then
    ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
    dpkg -i $ML_REPO_PKG
  fi

  # update package lists
  apt-get -y update

  # install packages
  CUDA_PKG_VERSION="7-5"
  CUDA_VERSION="7.5"
  apt-get install -y --no-install-recommends \
    cuda-core-$CUDA_PKG_VERSION \
    cuda-cudart-dev-$CUDA_PKG_VERSION \
    cuda-cublas-dev-$CUDA_PKG_VERSION \
    cuda-curand-dev-$CUDA_PKG_VERSION
  # manually create CUDA symlink
  ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

  if $WITH_CUDNN ; then
    apt-get install -y --no-install-recommends libcudnn5-dev
  fi
fi

