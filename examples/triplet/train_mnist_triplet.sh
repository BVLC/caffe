#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/triplet/mnist_triplet_solver.prototxt
