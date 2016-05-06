#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/mnist/mnist_rbm_solver.prototxt

