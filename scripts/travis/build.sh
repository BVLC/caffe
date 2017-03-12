#!/bin/bash
# build the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

cd build
make --jobs $NUM_THREADS all test.testbin
make lint
