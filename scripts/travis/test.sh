#!/bin/bash
# test the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if $WITH_CUDA ; then
  echo "Skipping tests for CUDA build"
  exit 0
fi

cd build
make runtest
make pytest
