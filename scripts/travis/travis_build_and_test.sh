#!/bin/bash
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if $WITH_CMAKE; then
  mkdir -p build
  cd build
  CPU_ONLY=" -DCPU_ONLY=ON"
  if ! $WITH_CUDA; then
    CPU_ONLY=" -DCPU_ONLY=OFF"
  fi

  if $WITH_CUDNN; then
    CUDNN_ARGS=" -DUSE_CUDNN=ON "
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
  cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $CUDNN_ARGS $PYTHON_ARGS -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" $IO_ARGS ..
  $MAKE
  $MAKE pytest
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
  if $WITH_IO; then
    export USE_LMDB=1
    export USE_LEVELDB=1
    export USE_OPENCV=1
  fi
  $MAKE all test pycaffe warn lint || true
  if ! $WITH_CUDA; then
    $MAKE runtest
  fi
  $MAKE all
  $MAKE test
  $MAKE pycaffe
  $MAKE pytest
  $MAKE warn
  if ! $WITH_CUDA; then
    $MAKE lint
  fi
fi
