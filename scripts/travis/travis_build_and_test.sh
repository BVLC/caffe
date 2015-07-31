#!/bin/bash
# Script called by Travis to do a CPU-only build of and test Caffe.

set -e

function w_CUDA() {
  if [ -n "$WITH_CUDA" ]
  then
    echo "CUDA is enabled.";
    return 0;
  else 
    echo "CUDA is disabled.";
    return 1;
  fi
}

function w_CMAKE() {
  if [ -n "$WITH_CMAKE" ]
  then
    echo "CMAKE is enabled.";
    return 0;
  else 
    echo "CMAKE is disabled.";
    return 1;
  fi
}

function call_make() {
  echo "Build with Makefile";
  if ! w_CUDA
  then
    export CPU_ONLY=1
  fi
  TARGETS=(all test pycaffe warn lint);
  for target in ${TARGETS[@]}; do
    echo "$MAKE $target ...";
    $MAKE $target;
    if [ $? != 0 ]; then
      echo "FAILED.";
      exit 1;
    fi
  done
  
  if ! w_CUDA
  then
    $MAKE runtest
  fi
}

function call_cmake() {
  echo "Build using CMAKE";  
  mkdir -p build
  cd build
  CPU_ONLY=" -DCPU_ONLY=ON"
  if ! w_CUDA
  then
    CPU_ONLY=" -DCPU_ONLY=OFF"
  fi
  PYTHON_ARGS=""
  if [ "$PYTHON_VERSION" = "3" ]; then
    PYTHON_ARGS="$PYTHON_ARGS -Dpython_version=3 -DBOOST_LIBRARYDIR=$CONDA_DIR/lib/"
  fi
  echo "execute 'cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" ..'";
  cmake -DBUILD_python=ON -DCMAKE_BUILD_TYPE=Release $CPU_ONLY $PYTHON_ARGS -DCMAKE_INCLUDE_PATH="$CONDA_DIR/include/" -DCMAKE_LIBRARY_PATH="$CONDA_DIR/lib/" ..
  $MAKE
  #$MAKE pytest
  if ! w_CUDA
  then
    $MAKE runtest
    $MAKE lint
  fi
  $MAKE clean
  cd - 
}

if [ -z "${NUM_THREADS}" ]; then
	export NUM_THREADS=1;
fi

MAKE="make --jobs=$NUM_THREADS --keep-going"

if w_CMAKE
then
  call_cmake  
else
  call_make
fi