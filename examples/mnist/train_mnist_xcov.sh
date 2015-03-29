#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/mnist/mnist_xcov_solver.prototxt
