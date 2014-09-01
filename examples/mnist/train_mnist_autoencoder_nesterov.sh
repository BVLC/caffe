#!/bin/bash

./build/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_nesterov.prototxt
