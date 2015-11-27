#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/siamese/mnist_siamese_solver.prototxt
