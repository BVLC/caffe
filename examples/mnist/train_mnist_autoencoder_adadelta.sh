#!/bin/bash

# Check if TOOLS is set
if [ -z ${TOOLS+x} ];
# if unset
then
  TOOLS=./build/tools
fi

$TOOLS/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_adadelta.prototxt
