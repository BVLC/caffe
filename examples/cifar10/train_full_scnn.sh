#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_scnn.prototxt \
    --weights=examples/cifar10/eilab_cifar10_full_ini_sparsenet.caffemodel

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1_scnn.prototxt \
    --snapshot=examples/cifar10/cifar10_full_scnn_iter_60000.solverstate

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2_scnn.prototxt \
    --snapshot=examples/cifar10/cifar10_full_scnn_iter_65000.solverstate
