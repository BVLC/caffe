#!/bin/bash
# Script called by Travis to do a CPU-only build of and test Caffe.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if $WITH_CMAKE; then
  mkdir build
  cd build
  cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON ..
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
  $MAKE all test pycaffe warn lint || true
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
