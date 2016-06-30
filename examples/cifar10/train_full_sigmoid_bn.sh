#!/usr/bin/env sh

# Check if TOOLS is unset
if [ -z "$TOOLS" ]; then
TOOLS=./build/tools
fi

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_sigmoid_solver_bn.prototxt

