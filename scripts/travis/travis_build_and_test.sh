#!/bin/bash
# Script called by Travis to do a CPU-only build of and test Caffe.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if $WITH_CMAKE; then
  mkdir build
  cd build
  if ! $WITH_CUDA; then
    export WITH_CUDA=OFF
  else
    export WITH_CUDA=ON
  fi
  cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=$WITH_CUDA -DWITH_HDF5=ON -DWITH_LEVELDB=ON \
    -DWITH_LMDB=ON ..
  $MAKE
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
  WITH_HDF5=1 WITH_LEVELDB=1 WITH_LMDB=1 $MAKE all test pycaffe warn lint || true
  if ! $WITH_CUDA; then
    $MAKE runtest
  fi
  $MAKE all
  $MAKE test
  $MAKE pycaffe
  $MAKE warn
  if ! $WITH_CUDA; then
    $MAKE lint
  fi
fi
