#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/mnist/mnist_kSAE_init_solver.prototxt

./build/tools/caffe train \
  --solver=examples/mnist/mnist_kSAE_init_solver.prototxt -- weights=examples/mnist/mnist_kSAE_init_iter_20000
