#!/bin/bash
# Script called by Travis to do a full build of Caffe,
# including CUDA functionality.

if $WITH_CMAKE; then
  mkdir build
  cd build
  cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..
  make --keep-going
  cd -
else
  export CPU_ONLY=0
  make --keep-going all test pycaffe warn
  make all
  make test
  make pycaffe
  make warn
fi
