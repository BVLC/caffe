#!/usr/bin/env sh
set -e

./build/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver.prototxt $@
