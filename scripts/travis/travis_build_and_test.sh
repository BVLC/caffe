#!/bin/bash
<<<<<<< HEAD
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.
=======
<<<<<<< HEAD
<<<<<<< HEAD
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.
=======
# Script called by Travis to do a CPU-only build of and test Caffe.
>>>>>>> origin/BVLC/parallel
=======
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if $WITH_CMAKE; then
  mkdir build
  cd build
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  CPU_ONLY=" -DCPU_ONLY=ON"
  if ! $WITH_CUDA; then
    CPU_ONLY=" -DCPU_ONLY=OFF"
  fi
  PYTHON_ARGS=""
  if [ "$PYTHON_VERSION" = "3" ]; then
    PYTHON_ARGS="$PYTHON_ARGS -Dpython_version=3 -DBOOST_LIBRARYDIR=$CONDA_DIR/lib/"
  fi
  if $WITH_IO; then
    IO_ARGS="-DUSE_OPENCV=ON -DUSE_LMDB=ON -DUSE_LEVELDB=ON"
  else
    IO_ARGS="-DUSE_OPENCV=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF"
  fi
  cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" $IO_ARGS ..
  $MAKE
  $MAKE pytest
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
  cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON ..
  $MAKE
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  if ! $WITH_CUDA; then
    $MAKE runtest
    $MAKE lint
  fi
  $MAKE clean
  cd -
else
  if ! $WITH_CUDA; then
    export CPU_ONLY=1
  fi
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  if $WITH_IO; then
    export USE_LMDB=1
    export USE_LEVELDB=1
    export USE_OPENCV=1
  fi
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  $MAKE all test pycaffe warn lint || true
  if ! $WITH_CUDA; then
    $MAKE runtest
  fi
  $MAKE all
  $MAKE test
  $MAKE pycaffe
<<<<<<< HEAD
  $MAKE pytest
=======
<<<<<<< HEAD
<<<<<<< HEAD
  $MAKE pytest
=======
>>>>>>> origin/BVLC/parallel
=======
  $MAKE pytest
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  $MAKE warn
  if ! $WITH_CUDA; then
    $MAKE lint
  fi
fi
