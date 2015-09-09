#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/cifar10/cifar10_cm_solver.prototxt --gpu=all 2>&1 | tee examples/cifar10/cifar10_cm_log.txt

