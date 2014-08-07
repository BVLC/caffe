#!/usr/bin/env sh

TOOLS=../../build/tools

$TOOLS/caffe.bin train --solver=cifar10_quick_solver.prototxt

# reduce learning rate by fctor of 10 after 8 epochs
$TOOLS/caffe.bin train \
  --solver=cifar10_quick_solver_lr1.prototxt \
  --snapshot=cifar10_quick_iter_4000.solverstate
