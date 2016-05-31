#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_CMAKE ; then
  make --jobs $NUM_THREADS all test pycaffe warn
else
  cd build
  make --jobs $NUM_THREADS all test.testbin
fi
make lint
