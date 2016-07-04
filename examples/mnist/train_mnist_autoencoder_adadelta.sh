#!/bin/bash

# Check if CAFFE_BIN is unset
if [ -z "$CAFFE_BIN" ]; then
  TOOLS=./build/tools
else
  TOOLS=$CAFFE_BIN
fi

$TOOLS/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_adadelta.prototxt
