#!/bin/bash
# Script called by Travis to do a CPU-only build of and test Caffe.

set -e
# Limit jobs to stay within available RAM/Swap
MAKE="make --jobs=2 --keep-going"

if $WITH_CMAKE; then
  mkdir build
  cd build
  cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON ..
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
  $MAKE pytest
  $MAKE warn
  if ! $WITH_CUDA; then
    $MAKE lint
  fi
fi
