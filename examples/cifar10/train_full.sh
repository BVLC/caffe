#!/usr/bin/env sh

# Check if TOOLS is unset
if [ -z ${TOOLS+x} ];
then
TOOLS=./build/tools
fi

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
