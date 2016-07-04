#!/usr/bin/env sh

# Check if $CAFFE_BIN is unset
if [ -z "$CAFFE_BIN" ]; then
  TOOLS=./build/tools
else
  TOOLS=$CAFFE_BIN
fi

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_sigmoid_solver_bn.prototxt

