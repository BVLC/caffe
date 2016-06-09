#!/usr/bin/env sh

# Check if TOOLS is unset
if [ -z ${TOOLS+x} ];
then
TOOLS=./build/tools
fi

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_sigmoid_solver.prototxt

